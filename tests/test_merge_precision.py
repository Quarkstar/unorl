"""Numerical checks that distinguish merge rounding from policy drift."""

import pytest
import torch

from unorl.merge_precision import (
    adam_step_counters,
    aggregate_output_shifts,
    output_shift,
    weight_rounding_metrics,
)


def test_adam_counters_accept_frozen_placeholders_but_reject_missing_history():
    state = {"step": torch.tensor(40), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.ones(2)}
    assert adam_step_counters([{}, state, {}]) == [40]
    with pytest.raises(ValueError, match="missing"):
        adam_step_counters([{"exp_avg": torch.zeros(2)}])


def test_small_update_can_disappear_in_bf16_without_disappearing_in_fp32():
    base = torch.ones(2, 3)
    a = torch.full((1, 3), 0.001)
    b = torch.ones(2, 1)
    row = weight_rounding_metrics(base, a, b, 1)
    assert row["changed_elements_lost_in_bf16"] == 6
    assert row["fp32_merge_error_energy"] < row["adapter_energy"] * 1e-7
    assert row["bf16_total_error_energy"] == pytest.approx(row["adapter_energy"])


def test_constant_logit_offset_changes_logits_but_preserves_policy():
    logits = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]])
    row = output_shift(logits, logits + 2, torch.tensor([1, 0]))
    assert row["kl_mean"] == 0
    assert row["chosen_logprob_mean_abs_shift"] == 0
    assert row["argmax_flip_fraction"] == 0
    assert row["logit_mean_abs_shift"] == 2


def test_summary_weights_tokens_and_preserves_worst_case():
    a = {"response_tokens": 1, "kl_mean": 0.1, "logit_rms_shift": 1.0, "logit_max_abs_shift": 5.0}
    b = {"response_tokens": 3, "kl_mean": 0.3, "logit_rms_shift": 3.0, "logit_max_abs_shift": 4.0}
    row = aggregate_output_shifts([a, b])
    assert row["kl_mean"] == pytest.approx(0.25)
    assert row["logit_rms_shift"] == pytest.approx(7**0.5)
    assert row["logit_max_abs_shift"] == 5
