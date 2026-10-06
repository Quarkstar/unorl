"""Shared-boundary ordinary ReLoRA with bounded effective weight updates."""

import sys

import ray
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from unorl.guided_reset_train import GuidedResetTrainer
from unorl.relora_train import ReLoRAExperiment
from unorl.train import MetricsCallback, register_math_environments
from unorl.update_cap_config import UpdateCapTrainConfig, validate_update_cap


class UpdateCapTrainer(GuidedResetTrainer):
    def build_models(self, policy_worker, critic_worker, ref_worker):
        from unorl.update_cap_worker import PolicyWorker

        return RayPPOTrainer.build_models(self, PolicyWorker, critic_worker, ref_worker)


class UpdateCapExperiment(ReLoRAExperiment):
    def get_trainer(self, **kwargs):
        trainer = UpdateCapTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    register_math_environments()
    UpdateCapExperiment(cfg).run()


def main():
    cfg = UpdateCapTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_update_cap(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
