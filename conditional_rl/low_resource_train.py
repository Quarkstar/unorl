"""SkyRL entrypoint for single-rollout signed REINFORCE with rank-one ReLoRA."""

import sys

import ray
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    AdvantageEstimatorRegistry,
    PolicyLossRegistry,
)
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from conditional_rl.low_resource import reinforce_loss, signed_returns
from conditional_rl.low_resource_config import LowResourceTrainConfig, validate_low_resource
from conditional_rl.train import BaselineExperiment, BenchmarkTrainer, MetricsCallback


def signed_advantage(token_level_rewards, response_mask, **kwargs):
    returns = signed_returns(token_level_rewards, response_mask)
    return returns, returns.clone()


def register_algorithms():
    PolicyLossRegistry.register("signed_reinforce", reinforce_loss)
    AdvantageEstimatorRegistry.register("signed_reinforce", signed_advantage)


class ReinforceTrainer(BenchmarkTrainer):
    def build_models(self, policy_worker, critic_worker, ref_worker):
        from conditional_rl.low_resource_worker import PolicyWorker

        return super().build_models(PolicyWorker, critic_worker, ref_worker)

    def _skip_policy_forward(self, training_input):
        return True  # REINFORCE does not consume old log probabilities.

    def _normalize_advantages(self, data, mini_batch_boundaries, prompt_boundaries=None):
        # Mean across trajectories, SUM across response tokens. Do not divide
        # by each trajectory's length or whiten rewards.
        advantages = data["advantages"].clone()
        for start, end in mini_batch_boundaries:
            advantages[start:end] /= end - start
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
