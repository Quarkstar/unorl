"""Low-memory training-gradient sketches for fresh rank-one ReLoRA cycles."""

import torch


@torch.no_grad()
def update_spaces(factors, relative_tolerance=1e-6):
    """Actual input/output singular spaces of a sum of low-rank updates.

    Thin QR plus a small SVD accounts for cancellation between cycles. A and
    B need not be individually orthogonal. No dense weight update is formed.
    """
    if not factors:
        raise ValueError("Need at least one accumulated update factor pair")
    b = torch.cat([bb.double() for _, bb in factors], dim=1)
    at = torch.cat([aa.double() for aa, _ in factors], dim=0).T
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(at, mode="reduced")
    u, values, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite accumulated spectrum")
    keep = values > values.max() * relative_tolerance
    return (qb @ u[:, keep]).float(), (qa @ vh[keep].T).float(), values


@torch.no_grad()
def fresh_probes(input_space, dimension, width, seed, device):
    """Orthonormal random input directions outside the accumulated input space."""
    width = min(width, dimension - input_space.shape[1])
    if width <= 0:
        raise ValueError("The accumulated update already spans the entire input space")
    generator = torch.Generator().manual_seed(seed)
    probes = torch.randn(dimension, width, generator=generator, dtype=torch.float64).to(device)
    q = input_space.to(device=device, dtype=torch.float64)
    # Twice projecting limits roundoff when existing directions dominate.
    for _ in range(2):
        probes -= q @ (q.T @ probes)
    return torch.linalg.qr(probes, mode="reduced").Q.float()


class GradientSketch:
    """Accumulate G @ probes from linear-layer input/output gradients.

    G is the actual un-clipped loss gradient. Sum microbatches before averaging
    across data-parallel ranks. Persistent storage is output_dim * probe_width;
    chunking bounds the temporary projected token matrix.
    """

    def __init__(self, probes, output_dimension, chunk_tokens=256):
        self.probes = probes
        self.gradient = torch.zeros(
            output_dimension, probes.shape[1], device=probes.device, dtype=torch.float32
        )
        self.chunk_tokens = chunk_tokens
        self.observations = 0

    @torch.no_grad()
    def accumulate(self, inputs, output_gradient):
        x = inputs.detach().reshape(-1, inputs.shape[-1])
        dy = output_gradient.detach().reshape(-1, output_gradient.shape[-1])
        if x.shape[0] != dy.shape[0]:
            raise ValueError("Input and output-gradient token counts differ")
        for start in range(0, x.shape[0], self.chunk_tokens):
            end = start + self.chunk_tokens
            projected = x[start:end].float() @ self.probes
            self.gradient.addmm_(dy[start:end].float().T, projected)
        self.observations += 1


@torch.no_grad()
def choose_direction(sketch, output_space, reference_a):
    """Top singular input direction of the doubly projected gradient sketch.

    This is the optimal rank-one Frobenius approximation within the random
    probe span, not an optimal Adam step or a guarantee of reward improvement.
    B starts at zero; its first gradient remains native (not output-projected).
    Match the reference Kaiming A norm to avoid changing adapter scale.
    """
    g = sketch.gradient.double()
    q = output_space.to(device=g.device, dtype=torch.float64)
    normal = g - q @ (q.T @ g)
    _, values, vh = torch.linalg.svd(normal, full_matrices=False)
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite normal-gradient sketch")
    energy = g.square().sum().item()
    normal_energy = normal.square().sum().item()
    diagnostics = {
        "sketch_l2": energy**0.5,
        "normal_sketch_l2": normal_energy**0.5,
        "normal_energy_fraction": normal_energy / max(energy, 1e-30),
        "leading_normal_singular_value": values[0].item(),
        "fallback_to_random": normal_energy <= max(energy * 1e-12, 1e-30),
    }
    if diagnostics["fallback_to_random"]:
        return reference_a.clone(), diagnostics
    direction = sketch.probes.double() @ vh[0]
    # SVD signs are arbitrary; fix the largest component positive.
    if direction[direction.abs().argmax()] < 0:
        direction.neg_()
    direction *= reference_a.double().norm() / direction.norm()
    return direction.reshape_as(reference_a).to(reference_a.dtype), diagnostics
