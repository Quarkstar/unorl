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
    current_a = a.clone()
    increment = math.radians(2)
    local_losses = []
    for _ in range(10):
        c1, c2 = (current_a * a).sum(), (current_a * q).sum()
        new_c1 = math.cos(increment) * c1 - math.sin(increment) * c2
        new_c2 = math.sin(increment) * c1 + math.cos(increment) * c2
        new_a = current_a + (new_c1 - c1) * a + (new_c2 - c2) * q
        torch.testing.assert_close(new_a.norm(), current_a.norm(), atol=1e-12, rtol=0)
        correction = b @ (current_a - new_a)
        torch.testing.assert_close(correction + b @ new_a, b @ current_a, atol=1e-12, rtol=0)
        _, residual = project_update(b, new_a, db @ current_a)
        local_losses.append(float(residual.norm() / (db.norm() * current_a.norm())))
        current_a = new_a
    expected = math.cos(math.radians(20)) * a + math.sin(math.radians(20)) * q
    torch.testing.assert_close(current_a, expected, atol=1e-12, rtol=0)
    # Two dense-gradient histories can be invisible to both old adapter factors
    # yet differ in the new B gradient. Therefore old Adam states alone cannot
    # determine exact state transport into the rotated adapter coordinates.
    hidden_gradient = db @ q
    old_a_gradient = b.T @ hidden_gradient
    old_b_gradient = hidden_gradient @ a.T
    torch.testing.assert_close(old_a_gradient, torch.zeros_like(a), atol=1e-12, rtol=0)
    torch.testing.assert_close(old_b_gradient, torch.zeros_like(b), atol=1e-12, rtol=0)
    new_b_gradient = hidden_gradient @ expected.T
    torch.testing.assert_close(
        new_b_gradient, math.sin(math.radians(20)) * db, atol=1e-12, rtol=0
    )
    report = {
        "scope": "Synthetic 11x13 rank-one float64 CPU example; scale absorbed in target update.",
        "seed": 42,
        "results": rows,
        "transition_example": {
            "increments": 10,
            "increment_degrees": 2,
            "intervening_training_updates": False,
            "maximum_local_affected_component_relative_residual": max(local_losses),
            "final_row_difference_from_one_20_degree_rotation": float(
                (current_a - expected).norm()
            ),
            "scope": "Checks fixed-plane composition and local compensation; not optimizer adaptation.",
        },
        "missing_history_counterexample": {
            "old_A_gradient_norm": float(old_a_gradient.norm()),
            "old_B_gradient_norm": float(old_b_gradient.norm()),
            "new_B_gradient_norm": float(new_b_gradient.norm()),
            "rotation_degrees": 20,
            "conclusion": (
                "Zero dense-gradient history and repeated hidden_gradient history have identical "
                "old adapter Adam states, but different first and second moments when their dense "
                "gradients are projected onto the rotated A. Exact counterfactual transport cannot "
                "be inferred solely from old adapter states. Small rotations reduce local change "
                "but do not remove this missing-information problem."
            ),
        },
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
