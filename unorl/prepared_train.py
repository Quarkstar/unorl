"""Matched GRPO with prepared, rank-aware adapter switches and native AdamW."""

import sys

import ray
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from unorl.prepared_config import PreparedTrainConfig, validate_prepared
from unorl.relora_train import ReLoRAExperiment, ReLoRATrainer
from unorl.train import MetricsCallback, register_math_environments


class PreparedTrainer(ReLoRATrainer):
    def build_models(self, policy_worker, critic_worker, ref_worker):
        from unorl.prepared_worker import PolicyWorker

        return RayPPOTrainer.build_models(self, PolicyWorker, critic_worker, ref_worker)


class PreparedExperiment(ReLoRAExperiment):
    def get_trainer(self, **kwargs):
        trainer = PreparedTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    register_math_environments()
    PreparedExperiment(cfg).run()


def main():
    cfg = PreparedTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_prepared(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
