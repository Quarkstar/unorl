"""Standard-LoRA GRPO with explicit ReLoRA merge and restart-ramp settings."""

from dataclasses import dataclass, field

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import TrainerConfig

from unorl.relora import restart_multiplier


@dataclass
class ReLoRATrainerConfig(TrainerConfig):
    relora_merge_interval: int = 40
    relora_first_merge_step: int | None = None
    relora_enable_merge: bool = True
    relora_restart_warmup_updates: int = 5
    relora_probe_response_tokens: int = 128


@dataclass
class ReLoRATrainConfig(SkyRLTrainConfig):
    trainer: ReLoRATrainerConfig = field(default_factory=ReLoRATrainerConfig)


def validate_relora(cfg):
    t, g = cfg.trainer, cfg.generator
    lora, algo, opt = t.policy.model.lora, t.algorithm, t.policy.optimizer_config
    restart_multiplier(
        0, t.relora_merge_interval, t.relora_restart_warmup_updates, t.relora_first_merge_step
    )
    conditions = {
        "standard full-layer rank-one LoRA": lora.rank == 1
        and lora.alpha == 32
        and lora.init_method == "kaiming"
        and lora.target_modules == "all-linear"
        and lora.exclude_modules is None
        and lora.dropout == 0,
        "native synchronous FSDP GRPO": t.strategy == "fsdp"
        and not t.policy.inference_only_init
        and algo.advantage_estimator == "grpo"
        and algo.policy_loss_type == "regular"
        and not t.fully_async.enabled,
        "constant base schedule without initial warmup/decay": opt.scheduler == "constant"
        and opt.num_warmup_steps == 0
        and opt.weight_decay == 0,
        "eight rollouts and one fresh-batch update": g.n_samples_per_prompt == 8
        and t.update_epochs_per_batch == 1
        and t.policy_mini_batch_size == t.train_batch_size,
        "no KL/critic": not algo.use_kl_loss
        and not algo.use_kl_in_reward
        and t.critic.model.path is None,
        "NCCL synchronization": g.inference_engine.weight_sync_backend == "nccl",
        "bounded response probe": 1 <= t.relora_probe_response_tokens <= 256,
    }
    failed = [name for name, ok in conditions.items() if not ok]
    if failed:
        raise ValueError("ReLoRA comparison requires: " + ", ".join(failed))
