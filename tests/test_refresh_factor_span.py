"""Verify compressed factor diagnostics against independent dense projections."""

import pytest
import torch

from scripts.research.analyze_refresh_factor_span import factor_geometry


@pytest.mark.parametrize("aligned_output", [False, True])
def test_factor_geometry_matches_dense_oracle(aligned_output):
    generator = torch.Generator().manual_seed(42)
    pairs = [
        (
            torch.randn(1, 11, generator=generator, dtype=torch.float64),
            torch.randn(13, 1, generator=generator, dtype=torch.float64),
        )
        for _ in range(3)
    ]
    if aligned_output:
        pairs = [(a, pairs[-1][1].clone()) for a, _ in pairs]
    result = factor_geometry(pairs)
    dense = sum(b @ a for a, b in pairs)
    energy = dense.square().sum()
    a, b = pairs[-1]
    u = b / b.norm()
    v = a.T / a.norm()
    output_residual = dense - u @ (u.T @ dense)
    input_residual = dense - (dense @ v) @ v.T
    singular = torch.linalg.svdvals(dense)
    assert result["energy_outside_active_output_direction"] == pytest.approx(
        (output_residual.square().sum() / energy).item(), abs=1e-12
    )
    assert result["energy_outside_active_input_direction"] == pytest.approx(
        (input_residual.square().sum() / energy).item(), abs=1e-12
    )
    assert result["accumulated_stable_rank"] == pytest.approx(
        (energy / singular[0].square()).item(), abs=1e-12
    )
    if aligned_output:
        assert result["energy_outside_active_output_direction"] < 1e-25
        assert result["accumulated_stable_rank"] == pytest.approx(1, abs=1e-12)
        assert result["normalized_input_direction_stable_rank"] > 1
