"""Check projection accuracy and prohibit dense model-sized diagnostic products."""

import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from unorl.refresh_geometry import update_space_residual


def test_factor_projection_matches_dense_projection_and_degenerate_adapters():
    torch.manual_seed(42)
    a, b = torch.randn(1, 13, dtype=torch.float64), torch.randn(11, 1, dtype=torch.float64)
    factors = [(torch.randn_like(a), torch.randn_like(b)) for _ in range(3)]
    dense = sum(column @ row for row, column in factors)
    for left, right in [(a, b), (torch.zeros_like(a), b), (a, torch.zeros_like(b))]:
        pa = (
            left.T @ left / left.square().sum()
            if left.norm()
            else torch.zeros(13, 13, dtype=a.dtype)
        )
        pb = (
            right @ right.T / right.square().sum()
            if right.norm()
            else torch.zeros(11, 11, dtype=a.dtype)
        )
        residual = (torch.eye(11, dtype=a.dtype) - pb) @ dense @ (torch.eye(13, dtype=a.dtype) - pa)
        result = update_space_residual(left, right, factors)
        assert result["target_l2"] == pytest.approx(float(dense.norm()), rel=1e-12)
        assert result["unavailable_l2"] == pytest.approx(float(residual.norm()), rel=1e-12)
    assert update_space_residual(a, b, []) == {
        "target_l2": 0,
        "unavailable_l2": 0,
        "relative_residual": 0,
    }


def test_large_projection_avoids_dense_matrix_and_obeys_rotation_sine():
    torch.manual_seed(17)
    a, b = torch.randn(1, 8192, dtype=torch.float64), torch.randn(6144, 1, dtype=torch.float64)
    a /= a.norm()
    b /= b.norm()
    q = torch.randn_like(a)
    q -= (q @ a.T) * a
    q /= q.norm()
    db = torch.randn_like(b)
    db -= b * (b.T @ db)
    new_a = math.cos(math.radians(20)) * a + math.sin(math.radians(20)) * q

    class RejectDenseProduct(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.mm.default:
                if (args[0].shape[0], args[1].shape[1]) == (6144, 8192):
                    raise AssertionError("Diagnostic attempted a dense weight-sized product")
            return func(*args, **(kwargs or {}))

    with RejectDenseProduct():
        result = update_space_residual(new_a, b, [(a, db)])
    assert result["relative_residual"] == pytest.approx(math.sin(math.radians(20)), rel=1e-12)
