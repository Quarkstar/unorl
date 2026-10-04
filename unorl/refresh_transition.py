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
