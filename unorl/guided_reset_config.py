"""Matched step-40 continuation branches for ordinary and guided ReLoRA."""

from dataclasses import dataclass, field

from unorl.relora_config import validate_relora
from unorl.relora_refresh_config import RefreshTrainConfig, RefreshTrainerConfig


@dataclass
class GuidedResetTrainerConfig(RefreshTrainerConfig):
    boundary_branch: str = "standard"
    boundary_probe_width: int = 8
    boundary_token_chunk: int = 256
    boundary_calibration_steps: list[int] = field(default_factory=lambda: [40, 80])
    boundary_validation_only: bool = False


@dataclass
class GuidedResetTrainConfig(RefreshTrainConfig):
    trainer: GuidedResetTrainerConfig = field(default_factory=GuidedResetTrainerConfig)


def validate_guided_reset(cfg):
    validate_relora(cfg)
    t = cfg.trainer
    if t.boundary_branch not in {"standard", "random", "guided"}:
        raise ValueError("Unknown boundary branch")
    if t.relora_enable_merge != (t.boundary_branch != "standard"):
        raise ValueError("Merge flag must match the selected branch")
    if t.relora_restart_warmup_updates or t.relora_refresh_angle_degrees:
        raise ValueError("Fresh resets use constant LR and no gradual refresh")
    target = 41 if t.boundary_validation_only else 100
    if t.boundary_calibration_steps != [40, 80] or t.max_training_steps != target:
        raise ValueError("The matched comparison must continue step 40 through global step 100")
    if t.boundary_validation_only and (
        t.boundary_branch != "guided" or t.eval_interval > 0 or t.ckpt_interval > 0
    ):
        raise ValueError("The native validation is one guided update without eval or checkpoints")
    if not 1 <= t.boundary_probe_width <= 32 or t.boundary_token_chunk <= 0:
        raise ValueError("Invalid gradient-sketch dimensions")
    if t.resume_mode != "from_path" or not str(t.resume_path).endswith("global_step_40"):
        raise ValueError("All branches must resume the shared pre-merge step-40 checkpoint")
