"""GRPO ReLoRA comparison with native AdamW and optional restart warmup."""

import sys

import ray
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from unorl.nora_merge_train import NoRAMergeTrainer
from unorl.relora_config import ReLoRATrainConfig, validate_relora
from unorl.train import BaselineExperiment, MetricsCallback, register_math_environments


class ReLoRATrainer(NoRAMergeTrainer):
    def build_models(self, policy_worker, critic_worker, ref_worker):
        from unorl.relora_worker import PolicyWorker

        return RayPPOTrainer.build_models(self, PolicyWorker, critic_worker, ref_worker)


class ReLoRAExperiment(BaselineExperiment):
    def get_trainer(self, **kwargs):
        trainer = ReLoRATrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    register_math_environments()
    ReLoRAExperiment(cfg).run()


def main():
    cfg = ReLoRATrainConfig.from_cli_overrides(sys.argv[1:])
    validate_relora(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
