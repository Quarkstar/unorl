from types import SimpleNamespace
from unittest.mock import Mock

import torch

from conditional_rl.ppo_train import ValueWarmupTrainer, monte_carlo_returns


def test_monte_carlo_returns_broadcast_terminal_outcomes_over_valid_tokens():
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0]])
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    returns = monte_carlo_returns(rewards, response_mask)

    expected = torch.tensor([[1.0, 1.0, 1.0, 0.0], [-1.0, -1.0, 0.0, 0.0]])
    torch.testing.assert_close(returns, expected)


def test_warmup_updates_critic_without_updating_policy():
    trainer = object.__new__(ValueWarmupTrainer)
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(critic_warmup_steps=5)
    )
    trainer.global_step = 1
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer._execute_training_step = Mock(return_value={"loss": 0.25})
    trainer.dispatch = Mock()
    batch = SimpleNamespace(metadata={})

    result = trainer.train_critic_and_policy(batch)

    assert result == {"loss": 0.25}
    trainer._execute_training_step.assert_called_once_with("critic", batch)
    assert trainer.all_metrics["ppo/value_warmup_active"] == 1.0
    assert trainer.all_metrics["ppo/policy_updates"] == 0.0
    trainer.dispatch.empty_cache.assert_called_once()
