"""Matched GRPO settings for a compensated rotation over several updates."""

from dataclasses import dataclass, field

from unorl.relora_config import ReLoRATrainConfig
from unorl.relora_refresh_config import RefreshTrainerConfig, validate_refresh


@dataclass
class GradualRefreshTrainerConfig(RefreshTrainerConfig):
    relora_refresh_updates: int = 10


@dataclass
class GradualRefreshTrainConfig(ReLoRATrainConfig):
    trainer: GradualRefreshTrainerConfig = field(default_factory=GradualRefreshTrainerConfig)


def validate_gradual_refresh(cfg):
    validate_refresh(cfg)
    trainer = cfg.trainer
    updates = trainer.relora_refresh_updates
    if type(updates) is not int or not 1 <= updates <= trainer.relora_merge_interval:
        raise ValueError("Refresh updates must be an integer from one to the merge interval")
    first = trainer.relora_first_merge_step
    first = trainer.relora_merge_interval if first is None else first
    relative = trainer.max_training_steps - first
    if (
        trainer.relora_enable_merge
        and relative >= 0
        and relative % trainer.relora_merge_interval < updates - 1
    ):
        raise ValueError("Training must not finish partway through a refresh transition")
