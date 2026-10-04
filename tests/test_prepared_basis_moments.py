"""Selected directions retain the exact observed projected-gradient moments."""

from scripts.research.diagnose_prepared_basis_moments import diagnose


def test_adaptive_basis_coefficients_match_native_adam_oracle():
    report = diagnose()
    assert report["coefficients_selected_after_history"]
    assert report["candidate_basis_width"] == 2
    assert report["max_native_adam_moment_error"] < 1e-12
    example = report["rank_one_stationary_example"]
    assert example["old_adapter_gradient_zero"]
    assert example["normal_weight_gradient_nonzero"]
    assert example["rank_one_loss"] == 0.5
    assert example["rank_two_loss"] == 0
