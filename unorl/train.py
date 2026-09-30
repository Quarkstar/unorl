"""GRPO entrypoint and shared SkyRL evaluation/metrics plumbing."""

import json
import sys
from pathlib import Path

import ray
import skyrl_gym
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.callbacks import TrainingCallback
from skyrl.train.utils.utils import validate_cfg

from unorl import benchmark_eval, grading  # noqa: F401
from unorl.prompts import system_tokenizer


def register_math_environments():
    """Install environments in the Ray entrypoint process, not only the driver."""
    for name, target in (
        ("benchmark_math", "unorl.benchmark_eval:BenchmarkEnv"),
        ("unorl_math", "unorl.grading:MathAnswerEnv"),
    ):
        if name not in skyrl_gym.registry:
            skyrl_gym.register(name, entry_point=target)


class MetricsCallback(TrainingCallback):
    def on_log(self, trainer, callback_input, control):
        self.write(trainer, "metrics.jsonl", callback_input.global_step, callback_input.logs)

    def on_eval_end(self, trainer, callback_input, control):
        if (
            not hasattr(trainer, "benchmark_batches")
            and trainer.cfg.generator.eval_n_samples_per_prompt == 8
        ):
            for key, value in list(callback_input.metrics.items()):
                if key.endswith("/mean_positive_reward"):
                    callback_input.metrics[key.replace("/mean_positive_reward", "/avg@8")] = value
        self.write(trainer, "evaluation.jsonl", callback_input.global_step, callback_input.metrics)

    @staticmethod
    def write(trainer, filename, step, metrics):
        path = Path(trainer.cfg.trainer.export_path).parent / filename
        with path.open("a") as handle:
            handle.write(json.dumps(dict(step=step, metrics=metrics), default=float) + "\n")


class BenchmarkTrainer(RayPPOTrainer):
    def build_models(self, policy_worker, critic_worker, ref_worker):
        if self.cfg.trainer.policy.model.lora.init_method == "lorafa":
            from unorl.lorafa_worker import PolicyWorker

            policy_worker = PolicyWorker
        elif self.cfg.trainer.policy.model.lora.init_method == "nora_init":
            from unorl.nora_worker import PolicyWorker

            policy_worker = PolicyWorker
        return super().build_models(policy_worker, critic_worker, ref_worker)

    async def eval(self, vllm_metrics_scraper=None):
        if not hasattr(self, "benchmark_batches"):
            self.benchmark_batches = {}
            expected_sizes = {"math500": 500, "amc23": 40, "aime25": 30, "aime26": 30}
            selected = {
                self.eval_dataset.dataframe[i]["data_source"] for i in range(len(self.eval_dataset))
            }
            assert selected and selected <= expected_sizes.keys(), selected
            for name, expected in expected_sizes.items():
                if name not in selected:
                    continue
                rows = [
                    self.eval_dataset[i]
                    for i in range(len(self.eval_dataset))
                    if self.eval_dataset.dataframe[i]["data_source"] == name
                ]
                assert len(rows) == expected, (name, len(rows))
                size = self.cfg.trainer.eval_batch_size
                self.benchmark_batches[name] = [
                    self.eval_dataset.collate_fn(rows[i : i + size])
                    for i in range(0, len(rows), size)
                ]
        old_loader = self.eval_dataloader
        n_samples = self.cfg.generator.eval_n_samples_per_prompt
        old_export = self.cfg.trainer.export_path
        metrics = {}
        try:
            for name, batches in self.benchmark_batches.items():
                self.eval_dataloader = batches
                self.cfg.trainer.export_path = str(Path(old_export) / name)
                current = await super().eval(vllm_metrics_scraper)
                metrics.update({k: v for k, v in current.items() if k.startswith(f"eval/{name}/")})
                if name == "math500":
                    metrics[f"eval/{name}/accuracy"] = current[f"eval/{name}/mean_positive_reward"]
                else:
                    metrics[f"eval/{name}/avg@{n_samples}"] = current[
                        f"eval/{name}/mean_positive_reward"
                    ]
                    metrics[f"eval/{name}/pass@{n_samples}"] = current[
                        f"eval/{name}/pass_at_{n_samples}"
                    ]
        finally:
            self.eval_dataloader = old_loader
            self.cfg.trainer.export_path = old_export
        return metrics

    async def train(self):
        try:
            await super().train()
        finally:
            await self.inference_engine_client.aclose()


class BaselineExperiment(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return super().get_generator(cfg, system_tokenizer(tokenizer), inference_engine_client)

    def get_trainer(self, **kwargs):
        trainer = BenchmarkTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    register_math_environments()
    BaselineExperiment(cfg).run()


def main():
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
