"""A controlled merge/reset extension of the full-layer NoRA-init GRPO recipe."""

from dataclasses import dataclass, field

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import TrainerConfig


@dataclass
class NoRAMergeTrainerConfig(TrainerConfig):
    nora_merge_interval: int = 40


@dataclass
class NoRAMergeTrainConfig(SkyRLTrainConfig):
    trainer: NoRAMergeTrainerConfig = field(default_factory=NoRAMergeTrainerConfig)


def validate_nora_merge(cfg):
    trainer, generator = cfg.trainer, cfg.generator
    algorithm = trainer.algorithm
    required = {
        "full-layer rank-one NoRA-init": trainer.policy.model.lora.rank == 1
        and trainer.policy.model.lora.alpha == 1
        and trainer.policy.model.lora.init_method == "nora_init"
        and trainer.policy.model.lora.target_modules == "all-linear"
        and trainer.policy.model.lora.exclude_modules is None,
        "positive merge interval": trainer.nora_merge_interval > 0,
        "native FSDP training": trainer.strategy == "fsdp"
        and not trainer.policy.inference_only_init,
        "synchronous GRPO": algorithm.advantage_estimator == "grpo"
        and algorithm.policy_loss_type == "regular"
        and not trainer.fully_async.enabled,
        "no KL or critic": not algorithm.use_kl_loss
        and not algorithm.use_kl_in_reward
        and trainer.critic.model.path is None,
        "constant LR without warmup": trainer.policy.optimizer_config.scheduler == "constant"
        and trainer.policy.optimizer_config.num_warmup_steps == 0,
        "eight rollouts": generator.n_samples_per_prompt == 8,
        "NCCL dense synchronization": generator.inference_engine.weight_sync_backend == "nccl",
        "one update per fresh batch": trainer.update_epochs_per_batch == 1,
    }
    failures = [name for name, valid in required.items() if not valid]
    if failures:
        raise ValueError("NoRA merge/reset requires: " + ", ".join(failures))
