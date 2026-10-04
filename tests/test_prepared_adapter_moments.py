"""Prepared projections recover information absent from native old moments."""

import torch

from scripts.research.diagnose_prepared_adapter_moments import diagnose


def test_old_projections_cannot_determine_new_direction():
    a = torch.tensor([[1.0, 0.0]])
    b = a.T
    invisible_gradient = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    zero_gradient = torch.zeros_like(invisible_gradient)
    for gradient in (zero_gradient, invisible_gradient):
        assert torch.count_nonzero(b.T @ gradient) == 0
        assert torch.count_nonzero(gradient @ a.T) == 0
    new_a = torch.tensor([[0.0, 1.0]])
    assert not torch.equal(zero_gradient @ new_a.T, invisible_gradient @ new_a.T)


def test_prepared_native_adam_moments_match_dense_observed_history():
    result = diagnose()
    assert result["observed_updates"] == 12
    assert result["installed_observer_step_counters"] == [12.0, 12.0]
    assert result["max_projected_gradient_error_vs_dense_oracle"] < 1e-12
    assert result["max_native_moment_error_vs_dense_oracle"] < 1e-12
    assert result["max_boundary_forward_error"] < 1e-12
    assert result["clipping_multiplier_range"][1] < 1
    assert result["observed_base_has_no_gradient"]
