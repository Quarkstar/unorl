"""PPO with a short Monte Carlo warmup for the value model."""

import sys
from dataclasses import dataclass, field

import ray
import torch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import TrainerConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import Timer, validate_cfg

from unorl.prompts import system_tokenizer
from unorl.train import BenchmarkTrainer, MetricsCallback


@dataclass
class ValueWarmupTrainerConfig(TrainerConfig):
    critic_warmup_steps: int = 5


@dataclass
class ValueWarmupPPOConfig(SkyRLTrainConfig):
    trainer: ValueWarmupTrainerConfig = field(default_factory=ValueWarmupTrainerConfig)


def monte_carlo_returns(rewards: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Compute undiscounted reward-to-go for terminal-outcome math rewards."""
    if rewards.shape != response_mask.shape:
        raise ValueError(f"Reward/mask shape mismatch: {rewards.shape} != {response_mask.shape}")
    masked_rewards = rewards * response_mask.to(rewards.dtype)
    return masked_rewards.flip(dims=(1,)).cumsum(dim=1).flip(dims=(1,)) * response_mask


class ValueWarmupTrainer(BenchmarkTrainer):
    """Fit the critic on on-policy Monte Carlo targets before enabling PPO updates."""

    @torch.no_grad()
    def compute_advantages_and_returns(self, data):
        if self.global_step > self.cfg.trainer.critic_warmup_steps:
            return super().compute_advantages_and_returns(data)

        returns = monte_carlo_returns(data["rewards"], data["response_mask"])
        data["returns"] = returns
        # The policy is frozen during warmup, so PPO advantages are not consumed.
        data["advantages"] = torch.zeros_like(returns)
        self.all_metrics["ppo/value_warmup_active"] = 1.0
        self.all_metrics["ppo/value_warmup_return_rms"] = torch.sqrt(
            returns.square().sum() / data["response_mask"].sum().clamp_min(1)
        ).item()
        return data

    def train_critic_and_policy(self, data):
        if self.global_step > self.cfg.trainer.critic_warmup_steps:
            self.all_metrics["ppo/value_warmup_active"] = 0.0
            self.all_metrics["ppo/policy_updates"] = float(
                self.global_step - self.cfg.trainer.critic_warmup_steps + 1
            )
            return super().train_critic_and_policy(data)

        data.metadata["global_step"] = self.global_step
        with Timer("critic_warmup", self.all_timings):
            critic_status = self._execute_training_step("critic", data)
        for key, value in critic_status.items():
            self.all_metrics[f"critic/{key}"] = value
        self.all_metrics["ppo/value_warmup_active"] = 1.0
        self.all_metrics["ppo/policy_updates"] = 0.0
        self.dispatch.empty_cache()
        return critic_status


class ValueWarmupPPOExperiment(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return super().get_generator(cfg, system_tokenizer(tokenizer), inference_engine_client)

    def get_trainer(self, **kwargs):
        trainer = ValueWarmupTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    ValueWarmupPPOExperiment(cfg).run()


def main():
    cfg = ValueWarmupPPOConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.trainer.algorithm.advantage_estimator != "gae":
        raise ValueError("PPO value warmup requires trainer.algorithm.advantage_estimator=gae")
    if not cfg.trainer.critic.model.path:
        raise ValueError("PPO value warmup requires trainer.critic.model.path")
    if cfg.trainer.critic_warmup_steps < 0:
        raise ValueError("trainer.critic_warmup_steps must be non-negative")
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
