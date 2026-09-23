"""Typed configuration for the low-resource SkyRL extension."""

from dataclasses import dataclass, field

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import BaseConfig, TrainerConfig


@dataclass
class LowResourceConfig(BaseConfig):
    lora_rank: int = 1
    lora_alpha: int = 1
    merge_interval: int = 10
    base_dtype: str = "float32"


@dataclass
class LowResourceTrainerConfig(TrainerConfig):
    low_resource: LowResourceConfig = field(default_factory=LowResourceConfig)


@dataclass
class LowResourceTrainConfig(SkyRLTrainConfig):
    trainer: LowResourceTrainerConfig = field(default_factory=LowResourceTrainerConfig)


def validate_low_resource(cfg):
    """Reject settings that silently change the on-policy estimator or sync."""
    trainer, generator = cfg.trainer, cfg.generator
    algorithm, resource = trainer.algorithm, trainer.low_resource
    required = {
        "rank-one LoRA": resource.lora_rank == 1,
        "positive LoRA scale": resource.lora_alpha > 0,
        "positive merge interval": resource.merge_interval > 0,
        "supported base dtype": resource.base_dtype in {"float32", "bfloat16"},
        "full-weight inference transport": trainer.policy.model.lora.rank == 0,
        "FSDP": trainer.strategy == "fsdp",
        "no critic": trainer.critic.model.path is None,
        "training enabled": not trainer.policy.inference_only_init,
        "no legacy importance correction": not algorithm.use_tis,
        "sequence-sum objective": algorithm.loss_reduction == "seq_mean_token_sum_norm"
        and algorithm.max_seq_len == 1,
        "synchronous training": not trainer.fully_async.enabled,
        "NCCL full-weight sync": generator.inference_engine.weight_sync_backend == "nccl",
        "one rollout per prompt": generator.n_samples_per_prompt == 1,
        "one update per fresh batch": trainer.update_epochs_per_batch == 1
        and trainer.policy_mini_batch_size == trainer.train_batch_size,
        "signed REINFORCE loss": algorithm.policy_loss_type == "signed_reinforce",
        "uncentered estimator": algorithm.advantage_estimator == "signed_reinforce"
        and not algorithm.advantage_batch_normalize,
        "no KL or entropy bonus": not algorithm.use_kl_loss
        and not algorithm.use_kl_in_reward
        and not algorithm.use_entropy_loss,
        "no filtering": not algorithm.zero_variance_filter
        and algorithm.dynamic_sampling.type is None
        and not generator.apply_overlong_filtering,
        "no importance correction": all(
            getattr(algorithm.off_policy_correction, name) is None
            for name in (
                "tis_ratio_type",
                "sequence_mask_metric",
                "outlier_token_is_threshold_low",
                "outlier_token_is_threshold_high",
                "token_mask_is_threshold_low",
                "token_mask_is_threshold_high",
            )
        ),
        "single-turn math": generator.batched
        and generator.max_turns == 1
        and not generator.step_wise_trajectories
        and cfg.environment.env_class == "conditional_math",
        "untruncated sampling distribution": generator.sampling_params.temperature == 1.0
        and generator.sampling_params.top_p == 1.0
        and generator.sampling_params.top_k == -1,
        "no weight decay": trainer.policy.optimizer_config.weight_decay == 0,
        "fixed learning rate without warmup": trainer.policy.optimizer_config.num_warmup_steps == 0
        and trainer.policy.optimizer_config.scheduler == "constant",
    }
    failures = [name for name, valid in required.items() if not valid]
    if failures:
        raise ValueError("Low-resource REINFORCE requires: " + ", ".join(failures))
