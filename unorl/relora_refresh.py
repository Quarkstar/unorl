"""Compensated gradual A refresh with warm B and native AdamW history."""

import math

import torch

from unorl.low_resource import adapter_layers, copy_parameter, full_tensor


def rotated_row(a, angle_degrees, generator):
    """Rotate a rank-one row toward a random orthogonal direction at equal norm."""
    if a.ndim != 2 or a.shape[0] != 1 or a.shape[1] < 2:
        raise ValueError("Refresh requires a rank-one row with at least two input features")
    if not 0 <= angle_degrees < 90:
        raise ValueError("Refresh angle must be in [0, 90) degrees")
    if angle_degrees == 0:
        return a.clone()
    norm = torch.linalg.vector_norm(a)
    if not torch.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot rotate a zero/nonfinite adapter row")
    direction = torch.randn(a.shape, dtype=a.dtype, device="cpu", generator=generator).to(a.device)
    direction -= (direction * a).sum() / norm.square() * a
    direction *= norm / torch.linalg.vector_norm(direction)
    angle = math.radians(angle_degrees)
    result = math.cos(angle) * a + math.sin(angle) * direction
    return result * (norm / torch.linalg.vector_norm(result))


@torch.no_grad()
def refresh_adapter(model, optimizer, seed, angle_degrees, on_correction=None):
    """Change A gradually and compensate in W; retain B, Adam variance/counters.

    W_new = W_old + scale * B @ (A_old - A_new), B_new = B_old.
    Hence W_new + scale * B_new @ A_new equals the old effective weight in
    exact arithmetic. The A gradient is unchanged on the same trajectory;
    the B gradient changes with the rotated input direction.

    Project B's first moment by cos(angle); retain its variance as a
    heuristic estimate for the unseen direction; its true variance may be
    larger. This is approximate
    history transfer, not exact Adam coordinate transport. A's moments and
    all bias-correction counters are unchanged. No dense optimizer state.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("Gradual refresh requires native AdamW")
    if not 0 <= angle_degrees < 90:
        raise ValueError("Refresh angle must be in [0, 90) degrees")
    if any(group["weight_decay"] != 0 for group in optimizer.param_groups):
        raise ValueError("The continuity derivation assumes zero weight decay")
    energy = 0.0
    rounding = 0.0
    count = 0
    cosine = math.cos(math.radians(angle_degrees))
    for count, (name, layer) in enumerate(adapter_layers(model), start=1):
        a_param = layer.lora_A["default"].weight
        b_param = layer.lora_B["default"].weight
        weight = layer.get_base_layer().weight
        a = full_tensor(a_param).float()
        b = full_tensor(b_param).float()
        generator = torch.Generator(device="cpu").manual_seed(seed + count)
        fresh = rotated_row(a, angle_degrees, generator)
        difference = a - fresh
        scaled_b = b * layer.scaling["default"]
        correction = scaled_b @ difference
        base = full_tensor(weight).float()
        updated = (base + correction).to(weight.dtype)
        if not torch.isfinite(updated).all():
            raise FloatingPointError("Nonfinite gradual-refresh base weights")
        energy += correction.square().sum().item()
        rounding += (updated.float() - base - correction).square().sum().item()
        copy_parameter(weight, updated)
        copy_parameter(a_param, fresh)
        state = optimizer.state.get(b_param, {})
        if "exp_avg" in state:
            state["exp_avg"].mul_(cosine)
        if on_correction is not None:
            on_correction(name, difference, scaled_b)
    if not count:
        raise ValueError("No rank-one adapters to refresh")
    return {
        "relora/merged_layers": float(count),
        "relora/delta_l2": math.sqrt(energy),
        "relora/rounding_relative_l2": math.sqrt(rounding / max(energy, 1e-30)),
        "refresh/angle_degrees": float(angle_degrees),
        "refresh/b_first_moment_projection": cosine,
        "refresh/optimizer_state_entries": float(len(optimizer.state)),
    }


def update_factors(old_a, old_b, new_a, new_b, scale):
    """Represent the actual optimizer weight update as two rank-one terms."""
    return [(old_a, (new_b - old_b) * scale), (new_a - old_a, new_b * scale)]


def factor_inner(left, right):
    """Frobenius inner product without materializing either dense update."""
    return sum(
        (a1 * a2).sum().item() * (b1 * b2).sum().item() for a1, b1 in left for a2, b2 in right
    )
