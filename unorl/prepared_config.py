"""Matched GRPO recipe with fixed-window prepared adapter history."""

from dataclasses import dataclass, field

from unorl.relora_config import ReLoRATrainConfig, ReLoRATrainerConfig, validate_relora


@dataclass
class PreparedTrainerConfig(ReLoRATrainerConfig):
    prepared_window_updates: int = 20
    prepared_min_total_fraction: float = 0.9
    prepared_angles: int = 256
    prepared_token_chunk: int = 256
    relora_track_updates: bool = True


@dataclass
class PreparedTrainConfig(ReLoRATrainConfig):
    trainer: PreparedTrainerConfig = field(default_factory=PreparedTrainerConfig)


def validate_prepared(cfg):
    validate_relora(cfg)
    t = cfg.trainer
    first = t.relora_first_merge_step or t.relora_merge_interval
    conditions = {
        "enabled merges and constant LR without restart ramp": t.relora_enable_merge
        and t.relora_restart_warmup_updates == 0,
        "preparation after the initial native update": type(t.prepared_window_updates) is int
        and 1 <= t.prepared_window_updates < first
        and t.prepared_window_updates <= t.relora_merge_interval,
        "declared local descent constraint": 0 <= t.prepared_min_total_fraction <= 1,
        "angular grid including both exact axes": type(t.prepared_angles) is int
        and t.prepared_angles >= 4
        and t.prepared_angles % 4 == 0,
        "bounded projection chunks": type(t.prepared_token_chunk) is int
        and t.prepared_token_chunk > 0,
        "finite-update detection through native clipping": t.policy.optimizer_config.max_grad_norm
        > 0,
    }
    failed = [name for name, valid in conditions.items() if not valid]
    if failed:
        raise ValueError("Prepared-history comparison requires: " + ", ".join(failed))
