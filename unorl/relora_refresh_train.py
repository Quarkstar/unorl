"""Matched GRPO checkpoint continuation with optional gradual adapter refresh."""

import sys

import ray
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from unorl.relora_refresh_config import RefreshTrainConfig, validate_refresh
from unorl.relora_train import ReLoRAExperiment, ReLoRATrainer
from unorl.train import MetricsCallback, register_math_environments


class RefreshTrainer(ReLoRATrainer):
    def build_models(self, policy_worker, critic_worker, ref_worker):
        from unorl.relora_refresh_worker import PolicyWorker

        return RayPPOTrainer.build_models(self, PolicyWorker, critic_worker, ref_worker)


class RefreshExperiment(ReLoRAExperiment):
    def get_trainer(self, **kwargs):
        trainer = RefreshTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    register_math_environments()
    RefreshExperiment(cfg).run()


def main():
    cfg = RefreshTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_refresh(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
