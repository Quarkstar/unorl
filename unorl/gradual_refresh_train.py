"""Matched GRPO with compensated fixed-plane refresh across several updates."""

import sys

import ray
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from unorl.gradual_refresh_config import GradualRefreshTrainConfig, validate_gradual_refresh
from unorl.relora_train import ReLoRAExperiment, ReLoRATrainer
from unorl.train import MetricsCallback, register_math_environments


class GradualRefreshTrainer(ReLoRATrainer):
    def build_models(self, policy_worker, critic_worker, ref_worker):
        from unorl.gradual_refresh_worker import PolicyWorker

        return RayPPOTrainer.build_models(self, PolicyWorker, critic_worker, ref_worker)


class GradualRefreshExperiment(ReLoRAExperiment):
    def get_trainer(self, **kwargs):
        trainer = GradualRefreshTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    register_math_environments()
    GradualRefreshExperiment(cfg).run()


def main():
    cfg = GradualRefreshTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_gradual_refresh(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
