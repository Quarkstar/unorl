"""Compensated fixed-plane increments; isolated from the running SkyRL worker."""

import math

import torch

from unorl.low_resource import adapter_layers, copy_parameter, full_tensor
from unorl.refresh_transition import rotate_in_refresh_plane


@torch.no_grad()
def apply_plane_refresh(model, optimizer, planes, angle_degrees, on_correction=None):
    """Apply one increment using caller-owned, checkpointed input planes.

    Preserve the effective weight in real arithmetic and keep B, A's Adam
    states, all second moments, and all counters. Scale B's first moment by
    the cosine of the actual row change. This is approximate history transfer,
    not reconstruction of previously unobserved dense gradients. The caller
    owns transition scheduling, correction history, and inference synchronization.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("Fixed-plane refresh requires native AdamW")
    if any(group["weight_decay"] != 0 for group in optimizer.param_groups):
        raise ValueError("The continuity derivation assumes zero weight decay")
    if not math.isfinite(angle_degrees) or not 0 <= angle_degrees < 90:
        raise ValueError("Angle must be finite and in [0, 90) degrees")
    layers = list(adapter_layers(model))
    if not layers or set(planes) != {name for name, _ in layers}:
        raise ValueError("Saved plane names must exactly match the active adapter layers")
    energy = rounding = 0.0
    cosine_min = 1.0
    for name, layer in layers:
        a_param = layer.lora_A["default"].weight
        b_param = layer.lora_B["default"].weight
        weight = layer.get_base_layer().weight
        # A must remain independent after copy_parameter updates the Parameter.
        a = full_tensor(a_param).float().clone()
        b = full_tensor(b_param).float()
        if a.ndim != 2 or a.shape[0] != 1 or b.ndim != 2 or b.shape[1] != 1:
            raise ValueError("Fixed-plane refresh requires rank-one adapters")
        plane = tuple(row.to(device=a.device, dtype=a.dtype) for row in planes[name])
        fresh = rotate_in_refresh_plane(a, plane, angle_degrees)
        norm_product = a.norm() * fresh.norm()
        if not torch.isfinite(norm_product) or norm_product <= 1e-24:
            raise ValueError("Cannot transport moments for a zero/nonfinite adapter row")
        cosine = (
            1.0 if angle_degrees == 0 else ((a * fresh).sum() / norm_product).clamp(-1, 1).item()
        )
        cosine_min = min(cosine_min, cosine)
        scale = layer.scaling["default"]
        correction = (b * scale) @ (a - fresh)
        base = full_tensor(weight).float()
        updated = (base + correction).to(weight.dtype)
        if not torch.isfinite(updated).all():
            raise FloatingPointError("Nonfinite fixed-plane base compensation")
        energy += correction.square().sum().item()
        rounding += (updated.float() - base - correction).square().sum().item()
        copy_parameter(weight, updated)
        copy_parameter(a_param, fresh)
        state = optimizer.state.get(b_param, {})
        if "exp_avg" in state:
            state["exp_avg"].mul_(cosine)
        if on_correction is not None:
            on_correction(name, a, fresh, b, plane, scale)
    metrics = {
        "relora/merged_layers": float(len(layers)),
        "relora/delta_l2": math.sqrt(energy),
        "relora/rounding_relative_l2": math.sqrt(rounding / max(energy, 1e-30)),
        "refresh/increment_angle_degrees": float(angle_degrees),
        "refresh/b_first_moment_projection_min": cosine_min,
        "refresh/optimizer_state_entries": float(len(optimizer.state)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Nonfinite fixed-plane refresh diagnostics")
    return metrics
