"""GRPO experiment with full-layer NoRA-init and periodic merge/reset."""

import sys

import ray
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from unorl.nora_merge_config import (
    NoRAMergeTrainConfig,
    validate_nora_merge,
)
from unorl.train import (
    BaselineExperiment,
    BenchmarkTrainer,
    MetricsCallback,
    register_math_environments,
)


class NoRAMergeTrainer(BenchmarkTrainer):
    def build_models(self, policy_worker, critic_worker, ref_worker):
        from unorl.nora_merge_worker import PolicyWorker

        # Bypass the ordinary NoRA worker selector; all other trainer behavior stays native.
        return RayPPOTrainer.build_models(self, PolicyWorker, critic_worker, ref_worker)

    def train_critic_and_policy(self, training_input):
        result = super().train_critic_and_policy(training_input)
        records = ray.get(
            self.policy_model.async_run_ray_method("pass_through", "resource_metrics")
        )
        self.all_metrics.update(records[0])
        return result


class NoRAMergeExperiment(BaselineExperiment):
    def get_trainer(self, **kwargs):
        trainer = NoRAMergeTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    register_math_environments()
    NoRAMergeExperiment(cfg).run()


def main():
    cfg = NoRAMergeTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_nora_merge(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
