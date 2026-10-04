"""CPU-verifiable fixed-plane rotation primitives; not wired into training."""

import math

import torch


@torch.no_grad()
def make_refresh_plane(a, seed):
    """Return two unit rows defining a reproducible plane for a rank-one A."""
    if a.ndim != 2 or a.shape[0] != 1 or a.shape[1] < 2:
        raise ValueError("Expected a rank-one adapter row with at least two features")
    if not a.is_floating_point() or not torch.isfinite(a).all():
        raise ValueError("Expected finite floating-point adapter values")
    norm = a.norm()
    if norm <= 1e-12:
        raise ValueError("Cannot define a refresh plane from a zero adapter")
    first = a.detach().clone() / norm
    generator = torch.Generator(device="cpu").manual_seed(seed)
    second = torch.randn(a.shape, dtype=a.dtype, device="cpu", generator=generator).to(a.device)
    second -= (second * first).sum() * first
    norm = second.norm()
    if not torch.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Sampled direction is degenerate")
    return first, second / norm


def rotate_in_refresh_plane(a, plane, angle_degrees):
    """Rotate current A in a saved plane, retaining any off-plane component.

    The plane is created once per transition and must be checkpointed. A can
    change between increments through ordinary optimizer updates. This pure
    function returns a row; callers must separately compensate base weights.
    It defines neither optimizer-state transport nor a merge schedule.
    """
    if not math.isfinite(angle_degrees) or not 0 <= angle_degrees < 90:
        raise ValueError("Angle must be finite and in [0, 90) degrees")
    first, second = plane
    if any(row.shape != a.shape or row.dtype != a.dtype or row.device != a.device for row in plane):
        raise ValueError("Plane rows must match the adapter shape, dtype, and device")
    if not torch.isfinite(a).all() or any(not torch.isfinite(row).all() for row in plane):
        raise ValueError("Adapter and plane must be finite")
    tolerance = 32 * torch.finfo(a.dtype).eps
    if (
        abs(first.square().sum().item() - 1) > tolerance
        or abs(second.square().sum().item() - 1) > tolerance
        or abs((first * second).sum().item()) > tolerance
    ):
        raise ValueError("Plane rows must be orthonormal")
    if angle_degrees == 0:
        return a.clone()
    angle = math.radians(angle_degrees)
    c1, c2 = (a * first).sum(), (a * second).sum()
    new_c1 = math.cos(angle) * c1 - math.sin(angle) * c2
    new_c2 = math.sin(angle) * c1 + math.cos(angle) * c2
    return a + (new_c1 - c1) * first + (new_c2 - c2) * second


@torch.no_grad()
def accumulate_plane_correction(columns, b, old_a, new_a, plane, scale):
    """Accumulate a cycle's base corrections as two columns without dense W.

    Every compensated difference lies in the same input plane. B may change
    between increments: sum(scale * B_t @ difference_t) still has at most two
    input directions. Return columns C1/C2 representing C1 @ e1 + C2 @ e2.
    This is diagnostic compression, not compression of active model weights.
    """
    if not math.isfinite(scale):
        raise ValueError("Scale must be finite")
    if b.ndim != 2 or b.shape[1] != 1 or not torch.isfinite(b).all():
        raise ValueError("Expected a finite rank-one output column")
    # Also validate saved plane invariants and shape/device/dtype compatibility.
    rotate_in_refresh_plane(old_a, plane, 0)
    if new_a.shape != old_a.shape or new_a.dtype != old_a.dtype or new_a.device != old_a.device:
        raise ValueError("Old and new rows must have matching shape, dtype, and device")
    if b.dtype != old_a.dtype or b.device != old_a.device or not torch.isfinite(new_a).all():
        raise ValueError("Column and rows must be finite and use the same dtype/device")
    difference = old_a - new_a
    coefficients = tuple((difference * unit).sum() for unit in plane)
    residual = difference - sum(
        coefficient * unit for coefficient, unit in zip(coefficients, plane)
    )
    tolerance = 128 * torch.finfo(old_a.dtype).eps * old_a.norm().clamp_min(1)
    if residual.norm() > tolerance:
        raise ValueError("Correction contains a direction outside the saved plane")
    if columns is None:
        columns = (torch.zeros_like(b), torch.zeros_like(b))
    if len(columns) != 2 or any(
        column.shape != b.shape or column.dtype != b.dtype or column.device != b.device
        for column in columns
    ):
        raise ValueError("Expected two compatible accumulated columns")
    result = tuple(
        column + scale * b * coefficient for column, coefficient in zip(columns, coefficients)
    )
    if any(not torch.isfinite(column).all() for column in result):
        raise FloatingPointError("Nonfinite accumulated correction")
    return result
