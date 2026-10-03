"""Matched checkpoint branches for gradual-refresh versus standard rank-one LoRA."""

from dataclasses import dataclass, field

from unorl.relora_config import ReLoRATrainConfig, ReLoRATrainerConfig, validate_relora


@dataclass
class RefreshTrainerConfig(ReLoRATrainerConfig):
    relora_refresh_angle_degrees: float = 20.0
    relora_track_updates: bool = True


@dataclass
class RefreshTrainConfig(ReLoRATrainConfig):
    trainer: RefreshTrainerConfig = field(default_factory=RefreshTrainerConfig)


def validate_refresh(cfg):
    validate_relora(cfg)
    t = cfg.trainer
    angle = t.relora_refresh_angle_degrees
    if (t.relora_enable_merge and not 0 < angle < 90) or (not t.relora_enable_merge and angle != 0):
        raise ValueError("Active refresh needs angle in (0, 90); the standard control uses zero")
    if t.relora_restart_warmup_updates != 0:
        raise ValueError("Gradual refresh retains constant LR and Adam counters")
