"""Rank-one LoFT-simple, specialized from the authors' MIT implementation.

Copyright (c) 2026 Nurbek Tastan. MIT notice: third_party/loft/LICENSE.
Reference: tnurbek/loft commit 148998d98f901cd7744db7baa5b2b4868fa16bf0.
Retain alternation, gradient calibration and first-moment transport. Second
moments remain elementwise without transport. No dense weight is constructed.
"""

import math

import torch
from peft.tuners.lora.layer import Linear
from torch.distributed.tensor import DTensor

from unorl.low_resource import full_tensor


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


class LoFTSimpleAdamW(torch.optim.Optimizer):
    """Adam-family LoFT-simple for one rank-one, unit-scale default adapter."""

    def __init__(self, model, lr, betas=(0.9, 0.999), eps=1e-4, weight_decay=0.0):
        if lr <= 0 or eps <= 0 or weight_decay != 0:
            raise ValueError("LoFT-simple requires positive LR/epsilon and zero weight decay")
        if not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid Adam betas")
        self.pairs = {}
        for layer in model.modules():
            if not isinstance(layer, Linear):
                continue
            if list(layer.lora_A) != ["default"] or layer.r["default"] != 1:
                raise ValueError("LoFT-simple requires one rank-one default adapter")
            if (
                layer.scaling["default"] != 1
                or layer.merged
                or layer.use_dora.get("default", False)
            ):
                raise ValueError("LoFT-simple requires unmerged ordinary LoRA with alpha/r=1")
            a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
            self.pairs[a] = (b, True)
            self.pairs[b] = (a, False)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params or set(params) != set(self.pairs):
            raise ValueError("Only paired A/B parameters may be trainable")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.update_A = False  # B first; initial B is zero.
        self.last_metrics = {}

    @torch.no_grad()
    def calibrated_grad_norm(self):
        """Authors' simple-variant clipping: active, rescaled factor gradients."""
        energy = 0.0
        for p, (other, is_a) in self.pairs.items():
            if p.grad is None or is_a != self.update_A:
                continue
            regularization = 1e-6 if is_a else 1e-8
            norm_sq = full_tensor(other).float().square().sum().item()
            grad = full_tensor(p.grad).float()
            energy += grad.square().sum().item() / (norm_sq + regularization) ** 2
        return math.sqrt(energy)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        gram_values, overlaps = [], []
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            # Preserve model parameter order, including the reference's within-step ordering.
            for p in group["params"]:
                if p.grad is None:
                    raise RuntimeError("LoFT-simple requires gradients for both factors")
                other, is_a = self.pairs[p]
                other_full = full_tensor(other).float()
                norm_sq = other_full.square().sum().item()
                inverse = 1.0 / (norm_sq + (1e-6 if is_a else 1e-8))
                grad = local_tensor(p.grad).float() * inverse
                if not torch.isfinite(grad).all():
                    raise FloatingPointError("Nonfinite calibrated LoFT gradient")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                moment = local_tensor(state["exp_avg"])
                previous = state.get("previous_other")
                if previous is not None:
                    dot = (other_full * previous.to(other_full.device)).sum().item()
                    moment.mul_(dot * inverse)
                    denom = math.sqrt(norm_sq * previous.float().square().sum().item())
                    if denom > 0:
                        overlaps.append(dot / denom)
                state["previous_other"] = other_full.clone()
                state["step"] += 1
                moment.mul_(beta1).add_(grad, alpha=1 - beta1)
                variance = local_tensor(state["exp_avg_sq"])
                variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                gram_values.append(norm_sq)
                if is_a != self.update_A:
                    continue
                denominator = (variance.sqrt() / math.sqrt(1 - beta2 ** state["step"])).add_(
                    group["eps"]
                )
                local_tensor(p).addcdiv_(
                    moment, denominator, value=-group["lr"] / (1 - beta1 ** state["step"])
                )
        self.last_metrics = {
            "loft/updated_A": float(self.update_A),
            "loft/opposite_factor_gram_min": min(gram_values),
            "loft/opposite_factor_gram_max": max(gram_values),
            "loft/mean_factor_cosine": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        }
        self.update_A = not self.update_A
        return loss

    def state_dict(self):
        state = super().state_dict()
        state["loft_update_A"] = self.update_A
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        self.update_A = state.pop("loft_update_A")
        super().load_state_dict(state)
