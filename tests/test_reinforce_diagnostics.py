"""Verify selection bias diagnostics and the negative gradient on truncated failures."""

from copy import deepcopy

import pytest
import torch
from skyrl.backends.skyrl_train.utils.ppo_utils import apply_loss_reduction_to_advantages_minibatch
from skyrl.train.config import SkyRLTrainConfig

from unorl.diagnostics import rollout_diagnostics
from unorl.reinforce_adamw_train import (
    batch_normalized_outcome_advantage,
    signed_reinforce_policy_loss,
)


def batch(filtered):
    return {
        "response_ids": [[1, 2], [3, 4], [5, 6, 7, 8]],
        "loss_masks": [[1, 1], [1, 1], [0 if filtered else 1] * 4],
        "rewards": [1.0, -1.0, -1.0],
        "stop_reasons": ["stop", "stop", "length"],
    }


def test_diagnostics_expose_mask_selection_without_mutating_rollouts():
    filtered = batch(True)
    original = deepcopy(filtered)
    metrics = rollout_diagnostics(filtered, normalize_rewards=True)
    assert filtered == original
    assert metrics["rollout/truncated_fraction"] == pytest.approx(1 / 3)
    assert metrics["rollout/masked_incorrect_fraction"] == 0.5
    assert metrics["advantage/active_response_mean"] > 0
    assert metrics["advantage/truncated_negative_mass_fraction"] == 0
    retained = rollout_diagnostics(batch(False), normalize_rewards=True)
    assert retained["advantage/active_response_mean"] == pytest.approx(0)
    assert retained["advantage/completed_response_mean"] > 0
    assert retained["advantage/truncated_negative_mass_fraction"] == pytest.approx(2 / 3)
    assert retained["rollout/trainable_token_fraction"] == 1


@pytest.mark.parametrize("filtered", [True, False])
def test_real_policy_loss_penalizes_truncated_failure_only_when_retained(filtered):
    cfg = SkyRLTrainConfig().trainer.algorithm
    cfg.off_policy_correction.tis_ratio_type = None
    response_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    mask = response_mask.clone()
    if filtered:
        mask[2] = 0
    rewards = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0]])
    advantages, _ = batch_normalized_outcome_advantage(rewards, response_mask)
    advantages = apply_loss_reduction_to_advantages_minibatch(
        advantages,
        mask,
        "token_mean",
        micro_batch_size=1,
        max_seq_len=4,
    )
    log_probs = torch.full((3, 4), -1.0, requires_grad=True)
    loss, _ = signed_reinforce_policy_loss(log_probs, log_probs.detach(), advantages, cfg, mask)
    loss.backward()
    assert (log_probs.grad[0, :2] < 0).all()
    assert (log_probs.grad[1, :2] > 0).all()
    assert (log_probs.grad[2] == 0).all() if filtered else (log_probs.grad[2] > 0).all()
    assert (log_probs.grad[:2, 2:] == 0).all()


def test_zero_variance_advantages_and_diagnostics_stay_finite():
    output = batch(False)
    output["rewards"] = [-1.0] * 3
    metrics = rollout_diagnostics(output, normalize_rewards=True)
    assert metrics["advantage/outcome_std"] == 0
    assert metrics["advantage/positive_token_mass_fraction"] == 0
    rewards = torch.tensor([[0.0, -1.0], [0.0, -1.0]])
    advantages, _ = batch_normalized_outcome_advantage(rewards, torch.ones_like(rewards))
    assert torch.equal(advantages, torch.zeros_like(advantages))
