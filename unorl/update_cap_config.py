"""One-factor intervention on the existing ordinary ReLoRA recipe."""

from dataclasses import dataclass, field

from unorl.guided_reset_config import (
    GuidedResetTrainConfig,
    GuidedResetTrainerConfig,
    validate_guided_reset,
)


@dataclass
class UpdateCapTrainerConfig(GuidedResetTrainerConfig):
    update_cap_budget_ratio: float = 1.0


@dataclass
class UpdateCapTrainConfig(GuidedResetTrainConfig):
    trainer: UpdateCapTrainerConfig = field(default_factory=UpdateCapTrainerConfig)


def validate_update_cap(cfg):
    validate_guided_reset(cfg)
    if cfg.trainer.boundary_branch != "random" or cfg.trainer.boundary_validation_only:
        raise ValueError("The update-cap trial uses the ordinary random-reset production branch")
    if cfg.trainer.update_cap_budget_ratio != 1.0:
        raise ValueError("First trial caps at exactly the preceding applied update norm")
