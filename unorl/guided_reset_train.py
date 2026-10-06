"""Matched GRPO branches from the shared pre-merge checkpoint."""

import sys

import ray
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from unorl.guided_reset_config import GuidedResetTrainConfig, validate_guided_reset
from unorl.relora_train import ReLoRAExperiment, ReLoRATrainer
from unorl.train import MetricsCallback, register_math_environments


class GuidedResetTrainer(ReLoRATrainer):
    def build_models(self, policy_worker, critic_worker, ref_worker):
        from unorl.guided_reset_worker import PolicyWorker

        return RayPPOTrainer.build_models(self, PolicyWorker, critic_worker, ref_worker)


class GuidedResetExperiment(ReLoRAExperiment):
    def get_trainer(self, **kwargs):
        trainer = GuidedResetTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    register_math_environments()
    GuidedResetExperiment(cfg).run()


def main():
    cfg = GuidedResetTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_guided_reset(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
