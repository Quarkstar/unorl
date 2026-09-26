"""SkyRL entrypoint for single-rollout signed REINFORCE with rank-one ReLoRA."""

import sys

import ray
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    AdvantageEstimatorRegistry,
    PolicyLossRegistry,
)
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from unorl.low_resource import reinforce_loss, signed_returns
from unorl.low_resource_config import LowResourceTrainConfig, validate_low_resource
from unorl.train import BaselineExperiment, BenchmarkTrainer, MetricsCallback


def signed_advantage(token_level_rewards, response_mask, **kwargs):
    returns = signed_returns(token_level_rewards, response_mask)
    return returns, returns.clone()


def batch_mean_advantage(token_level_rewards, response_mask, **kwargs):
    """Center signed outcome rewards by their current on-policy batch mean."""
    signed_returns(token_level_rewards, response_mask)
    scores = token_level_rewards.sum(dim=-1, keepdim=True)
    centered = (scores - scores.mean()) * response_mask.to(token_level_rewards.dtype)
    return centered, centered.clone()


def batch_normalized_advantage(token_level_rewards, response_mask, **kwargs):
    """Center and scale signed terminal rewards across the rollout batch."""
    signed_returns(token_level_rewards, response_mask)
    scores = token_level_rewards.sum(dim=-1, keepdim=True)
    centered = scores - scores.mean()
    scale = scores.std(unbiased=False).clamp_min(1e-8)
    normalized = (centered / scale) * response_mask.to(token_level_rewards.dtype)
    return normalized, normalized.clone()


def register_algorithms():
    PolicyLossRegistry.register("signed_reinforce", reinforce_loss)
    AdvantageEstimatorRegistry.register("signed_reinforce", signed_advantage)
    AdvantageEstimatorRegistry.register("batch_mean_reinforce", batch_mean_advantage)
    AdvantageEstimatorRegistry.register("batch_norm_reinforce", batch_normalized_advantage)


class ReinforceTrainer(BenchmarkTrainer):
    def build_models(self, policy_worker, critic_worker, ref_worker):
        from unorl.low_resource_worker import PolicyWorker

        return super().build_models(PolicyWorker, critic_worker, ref_worker)

    def _skip_policy_forward(self, training_input):
        return True  # REINFORCE does not consume old log probabilities.

    def _normalize_advantages(self, data, mini_batch_boundaries, prompt_boundaries=None):
        # Token mean over each policy minibatch. Each response's signed terminal
        # reward is broadcast over its response tokens; average over all valid
        # response tokens to match SkyRL's `token_mean` reduction.
        advantages = data["advantages"].clone()
        for start, end in mini_batch_boundaries:
            token_count = data["loss_mask"][start:end].sum().clamp(min=1)
            advantages[start:end] /= token_count
        data["advantages"] = advantages
        return data

    def train_critic_and_policy(self, training_input):
        status = super().train_critic_and_policy(training_input)
        metrics = ray.get(
            self.policy_model.async_run_ray_method("pass_through", "resource_metrics")
        )
        self.all_metrics.update(metrics[0])
        return status


class LowResourceExperiment(BaselineExperiment):
    def get_trainer(self, **kwargs):
        trainer = ReinforceTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    LowResourceExperiment(cfg).run()


def main():
    cfg = LowResourceTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_low_resource(cfg)
    initialize_ray(cfg)
    try:
        register_algorithms()
        validate_cfg(cfg)
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
