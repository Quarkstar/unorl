"""Verify refresh update-space geometry on a small float64 CPU example."""

import json
import math
from pathlib import Path

import torch


def project_update(b, a, target):
    """Project a weight update onto rank-one adapter's first-order update space."""
    da = b.T @ target / b.square().sum()
    db = (target @ a.T - b @ (da @ a.T)) / a.square().sum()
    projected = b @ da + db @ a
    residual = target - projected
    assert torch.allclose(b.T @ residual, torch.zeros_like(b.T @ residual), atol=1e-12)
    assert torch.allclose(residual @ a.T, torch.zeros_like(residual @ a.T), atol=1e-12)
    return projected, residual


def main():
    torch.manual_seed(42)
    b = torch.randn(11, 1, dtype=torch.float64)
    a = torch.randn(1, 13, dtype=torch.float64)
    b /= b.norm()
    a /= a.norm()
    q = torch.randn_like(a)
    q -= (q @ a.T) * a
    q /= q.norm()
    db = torch.randn_like(b)
    db -= b * (b.T @ db)
    db /= db.norm()
    da = 0.3 * torch.randn_like(a)
    pure = db @ a
    mixed = b @ da + pure
    rows = []
    for degrees in (0, 20, 45, 90):
        theta = math.radians(degrees)
        new_a = math.cos(theta) * a + math.sin(theta) * q
        compensation = b @ (a - new_a)
        assert torch.allclose(compensation + b @ new_a, b @ a, atol=1e-12)
        _, residual = project_update(b, new_a, pure)
        relative = float(residual.norm() / pure.norm())
        assert math.isclose(relative, math.sin(theta), abs_tol=1e-12)
        _, mixed_residual = project_update(b, new_a, mixed)
        rows.append(
            {
                "angle_degrees": degrees,
                "function_difference_max": float((compensation + b @ new_a - b @ a).abs().max()),
                "pure_orthogonal_B_update_relative_residual": relative,
                "expected_sine": math.sin(theta),
                "mixed_update_relative_residual": float(mixed_residual.norm() / mixed.norm()),
            }
        )
    report = {
        "scope": "Synthetic 11x13 rank-one float64 CPU example; scale absorbed in target update.",
        "seed": 42,
        "results": rows,
        "limitations": (
            "First-order parameter update geometry, not an Adam transport theorem or model performance "
            "measurement. Finite factor updates include the additional db @ da term. Compensating W "
            "preserves the function but does not preserve the adapter tangent space."
        ),
    }
    path = Path(__file__).resolve().parents[2] / "research/data/refresh-tangent-analysis.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
