"""Fixed per-cycle projections sufficient for adaptive rank-one Adam warm starts.

This module observes gradients only. It neither attaches model hooks nor
changes parameters, optimizer states, clipping rules or inference weights.
"""

import math

import torch


class PreparedBasisMoments:
    """Two candidate directions on each side, with packed raw cross moments.

    Call accumulate for each already loss-normalized microbatch. At the end
    of the update, reduce the accumulated projections across data-parallel
    ranks if necessary, then apply the observed clipping multiplier exactly
    once through finish_update. Bases stay fixed throughout this history.
    """

    _buffers = ("ma", "mb", "ca", "cb", "ga", "gb")

    def __init__(self, qin, qout, beta1=0.9, beta2=0.999, chunk_tokens=256):
        if any(not math.isfinite(b) or not 0 <= b < 1 for b in (beta1, beta2)):
            raise ValueError("EMA decay must be finite and in [0, 1)")
        if type(chunk_tokens) is not int or chunk_tokens <= 0:
            raise ValueError("Token chunk size must be a positive integer")
        if qin.dtype not in (torch.float32, torch.float64):
            raise ValueError("Projection state requires float32 or float64")
        if qin.dtype != qout.dtype or qin.device != qout.device:
            raise ValueError("Candidate bases must share dtype and device")
        for q in (qin, qout):
            if q.ndim != 2 or q.shape[1] != 2 or not torch.isfinite(q).all():
                raise ValueError("Expected a finite two-column candidate basis")
            eye = torch.eye(2, dtype=q.dtype, device=q.device)
            if not torch.allclose(q.T @ q, eye, atol=1e-5, rtol=0):
                raise ValueError("Candidate columns must be orthonormal")
        self.qin, self.qout = qin.detach().clone(), qout.detach().clone()
        self.beta1, self.beta2 = beta1, beta2
        self.chunk_tokens = chunk_tokens
        self.count = self.pending_microbatches = 0
        self.ma = qin.new_zeros(2, qin.shape[0])
        self.mb = qout.new_zeros(qout.shape[0], 2)
        # Packed covariance entries: [00, 01, 11].
        self.ca = qin.new_zeros(3, qin.shape[0])
        self.cb = qout.new_zeros(qout.shape[0], 3)
        self.ga, self.gb = torch.zeros_like(self.ma), torch.zeros_like(self.mb)

    @torch.no_grad()
    def accumulate(self, x, dy):
        """Project one microbatch without forming an out-by-in weight gradient."""
        if x.ndim not in (2, 3) or dy.ndim != x.ndim:
            raise ValueError("Expected two- or three-dimensional microbatch tensors")
        if x.shape[:-1] != dy.shape[:-1]:
            raise ValueError("Activation and output-gradient token shapes differ")
        if x.shape[-1] != self.qin.shape[0] or dy.shape[-1] != self.qout.shape[0]:
            raise ValueError("Layer dimensions differ from candidate bases")
        if x.device != self.qin.device or dy.device != self.qout.device:
            raise ValueError("Microbatch and observer devices differ")
        x, dy = x.detach(), dy.detach()
        blocks = ((x, dy),) if x.ndim == 2 else ((x[i], dy[i]) for i in range(x.shape[0]))
        # Cast only one bounded chunk; avoid a full FP32 activation copy.
        for inputs_block, outputs_block in blocks:
            for start in range(0, inputs_block.shape[0], self.chunk_tokens):
                inputs = inputs_block[start : start + self.chunk_tokens].to(self.qin.dtype)
                outputs = outputs_block[start : start + self.chunk_tokens].to(self.qout.dtype)
                self.gb.add_(outputs.T @ (inputs @ self.qin))
                self.ga.add_((outputs @ self.qout).T @ inputs)
        self.pending_microbatches += 1

    @staticmethod
    def _packed_square(value):
        return torch.stack((value[0].square(), value[0] * value[1], value[1].square()))

    @torch.no_grad()
    def finish_update(self, clip_multiplier):
        """Update moments after caller-owned reduction and observed clipping."""
        if not self.pending_microbatches:
            raise ValueError("Cannot update moments without any microbatch")
        if not math.isfinite(clip_multiplier) or not 0 <= clip_multiplier <= 1:
            raise ValueError("Clipping multiplier must be finite and in [0, 1]")
        if not torch.isfinite(self.ga).all() or not torch.isfinite(self.gb).all():
            raise FloatingPointError("Nonfinite projected gradient")
        self.ga.mul_(clip_multiplier)
        self.gb.mul_(clip_multiplier)
        self.ma.mul_(self.beta1).add_(self.ga, alpha=1 - self.beta1)
        self.mb.mul_(self.beta1).add_(self.gb, alpha=1 - self.beta1)
        self.ca.mul_(self.beta2).add_(self._packed_square(self.ga), alpha=1 - self.beta2)
        self.cb.mul_(self.beta2).add_(self._packed_square(self.gb.T).T, alpha=1 - self.beta2)
        self.count += 1
        self.pending_microbatches = 0
        self.ga.zero_()
        self.gb.zero_()

    @torch.no_grad()
    def select_coefficients(self, a_norm, b_norm):
        """Maximize observed mean-gradient alignment inside the candidate spans."""
        if self.pending_microbatches or not self.count:
            raise ValueError("Direction selection requires complete observed updates")
        if any(not math.isfinite(n) or n <= 0 for n in (a_norm, b_norm)):
            raise ValueError("Prepared factor norms must be finite and positive")
        core = (self.qout.T @ self.mb + self.ma @ self.qin) / 2
        if not torch.isfinite(core).all() or core.norm() == 0:
            raise ValueError("Candidate spans have no finite nonzero mean-gradient signal")
        left, _, right = torch.linalg.svd(core)
        return right[:1] * a_norm, left[:, :1] * b_norm

    @staticmethod
    def _quadratic(packed, coefficients):
        c0, c1 = coefficients.reshape(-1)
        terms = torch.stack(
            (packed[0] * c0.square(), 2 * packed[1] * c0 * c1, packed[2] * c1.square())
        )
        value = terms.sum(dim=0)
        tolerance = 32 * torch.finfo(value.dtype).eps * terms.abs().sum(dim=0)
        if (value < -tolerance).any():
            raise FloatingPointError("Cross moments yield a materially negative variance")
        return value.clamp_min(0)

    @torch.no_grad()
    def map_moments(self, a_coeff, b_coeff, scale):
        """Recover raw native moments for finally selected fixed coordinates.

        Returned step is the observation count, not the global training step.
        The moments have not been bias-corrected; native Adam does that itself.
        """
        if self.pending_microbatches or not self.count:
            raise ValueError("Moment mapping requires complete observed updates")
        if a_coeff.shape != (1, 2) or b_coeff.shape != (2, 1):
            raise ValueError("Rank-one coefficients require shapes [1,2] and [2,1]")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Adapter scale must be finite and positive")
        for c in (a_coeff, b_coeff):
            if (
                c.device != self.qin.device
                or c.dtype != self.qin.dtype
                or not torch.isfinite(c).all()
            ):
                raise ValueError("Coefficients must share the finite observer dtype/device")
        result = {
            "a": a_coeff @ self.qin.T,
            "b": self.qout @ b_coeff,
            "ma": scale * b_coeff.T @ self.ma,
            "mb": scale * self.mb @ a_coeff.T,
            "va": scale**2 * self._quadratic(self.ca, b_coeff).reshape(1, -1),
            "vb": scale**2 * self._quadratic(self.cb.T, a_coeff).reshape(-1, 1),
            "step": self.count,
        }
        if any(not torch.isfinite(v).all() for v in result.values() if isinstance(v, torch.Tensor)):
            raise FloatingPointError("Nonfinite mapped moments")
        return result

    def state_dict(self):
        """Own CPU copies, including an unfinished microbatch accumulation."""
        result = {
            "version": 1,
            "count": self.count,
            "pending_microbatches": self.pending_microbatches,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "chunk_tokens": self.chunk_tokens,
        }
        for name in ("qin", "qout", *self._buffers):
            result[name] = getattr(self, name).detach().cpu().clone()
        return result

    @property
    def resident_tensor_bytes(self):
        """Exact observer tensor storage, excluding temporary products/checkpoint copies."""
        return sum(
            getattr(self, name).numel() * getattr(self, name).element_size()
            for name in ("qin", "qout", *self._buffers)
        )

    @classmethod
    def from_state_dict(cls, state, device="cpu"):
        if type(state.get("version")) is not int or state["version"] != 1:
            raise ValueError("Unsupported prepared-moment checkpoint version")
        instance = cls(
            state["qin"].to(device),
            state["qout"].to(device),
            state["beta1"],
            state["beta2"],
            state["chunk_tokens"],
        )
        for name in ("count", "pending_microbatches"):
            value = state[name]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid observer update or microbatch count")
            setattr(instance, name, value)
        for name in cls._buffers:
            value = state[name]
            target = getattr(instance, name)
            if (
                value.shape != target.shape
                or value.dtype != target.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid prepared-moment checkpoint tensor")
            target.copy_(value.to(device))
        if not instance.pending_microbatches and (
            instance.ga.count_nonzero() or instance.gb.count_nonzero()
        ):
            raise ValueError("Completed observer checkpoint has pending gradient values")
        if (instance.ca[[0, 2]] < 0).any() or (instance.cb[:, [0, 2]] < 0).any():
            raise ValueError("Negative diagonal second moments")
        for covariance in (instance.ca, instance.cb.T):
            c0, cross, c1 = covariance.double()
            product, squared_cross = c0 * c1, cross.square()
            tolerance = 32 * torch.finfo(covariance.dtype).eps * (product + squared_cross)
            if (squared_cross > product + tolerance).any():
                raise ValueError("Checkpoint cross moments are not positive semidefinite")
        if instance.count == 0 and any(
            getattr(instance, name).count_nonzero() for name in ("ma", "mb", "ca", "cb")
        ):
            raise ValueError("Unobserved history cannot have nonzero moments")
        return instance
