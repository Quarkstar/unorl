"""Standard SkyRL AdamW training for signed, single-rollout REINFORCE."""

import sys

import ray
import torch
from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import apply_off_policy_correction
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    AdvantageEstimatorRegistry,
    PolicyLossRegistry,
    reduce_loss,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg

from unorl.diagnostics import rollout_diagnostics
from unorl.train import BaselineExperiment, BenchmarkTrainer, MetricsCallback


def signed_outcome_advantage(token_level_rewards, response_mask, **kwargs):
    """Broadcast the uncentered signed terminal reward over response tokens."""
    scores = token_level_rewards.sum(dim=-1, keepdim=True)
    if not torch.isfinite(scores).all() or not ((scores == 1) | (scores == -1)).all():
        raise ValueError("Signed REINFORCE expects terminal rewards exactly +1 or -1")
    advantages = scores * response_mask.to(token_level_rewards.dtype)
    return advantages, advantages.clone()


def batch_normalized_outcome_advantage(token_level_rewards, response_mask, **kwargs):
    """Z-score signed outcome rewards across fresh on-policy trajectories."""
    scores = token_level_rewards.sum(dim=-1, keepdim=True)
    if not torch.isfinite(scores).all() or not ((scores == 1) | (scores == -1)).all():
        raise ValueError("Batch-normalized REINFORCE expects terminal rewards exactly +1 or -1")
    mean = scores.mean()
    std = scores.std(unbiased=False).clamp_min(1e-8)
    advantages = ((scores - mean) / std) * response_mask.to(token_level_rewards.dtype)
    return advantages, advantages.clone()


def signed_reinforce_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None
):
    """Direct REINFORCE objective with SkyRL's configured importance correction."""
    if loss_mask is None:
        loss_mask = torch.ones_like(log_probs)
    loss = -log_probs * advantages.detach()
    loss, loss_mask, correction_metrics = apply_off_policy_correction(
        loss, old_log_probs, rollout_logprobs, loss_mask, config.off_policy_correction
    )
    loss = reduce_loss(loss, loss_mask)
    return loss, {"clip_ratio": 0.0, **correction_metrics}


class SignedReinforceTrainer(BenchmarkTrainer):
    """Use default FSDP/AdamW with observable truncation and loss-mask selection."""

    def convert_to_training_input(self, generator_output, uids):
        self.all_metrics.update(
            rollout_diagnostics(
                generator_output,
                normalize_rewards=self.cfg.trainer.algorithm.advantage_estimator
                == "batch_norm_reinforce",
            )
        )
        return super().convert_to_training_input(generator_output, uids)


class SignedReinforceExperiment(BaselineExperiment):
    def get_trainer(self, **kwargs):
        trainer = SignedReinforceTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    SignedReinforceExperiment(cfg).run()


def validate_experiment_config(cfg):
    trainer, algorithm, generator = cfg.trainer, cfg.trainer.algorithm, cfg.generator
    expected = {
        "standard FSDP/AdamW path": trainer.strategy == "fsdp",
        "native rank-one LoRA": trainer.policy.model.lora.rank == 1,
        "single rollout": generator.n_samples_per_prompt == 1,
        "batch size 256": trainer.train_batch_size == 256,
        "one full-batch update": trainer.policy_mini_batch_size == trainer.train_batch_size
        and trainer.update_epochs_per_batch == 1,
        "signed REINFORCE loss": algorithm.policy_loss_type == "signed_reinforce",
        "supported advantage estimator": algorithm.advantage_estimator
        in {"signed_reinforce", "batch_norm_reinforce"},
        "no KL or entropy bonus": not algorithm.use_kl_loss
        and not algorithm.use_kl_in_reward
        and not algorithm.use_entropy_loss,
        "token-level TIS correction": algorithm.off_policy_correction.tis_ratio_type == "token",
        "explicit truncation policy": isinstance(generator.apply_overlong_filtering, bool),
        "preserve signed rewards on truncation": not generator.zero_reward_on_non_stop,
        "one normalization and token-mean reduction": not algorithm.advantage_batch_normalize
        and algorithm.loss_reduction == "token_mean",
        "constant LR without warmup": trainer.policy.optimizer_config.scheduler == "constant"
        and trainer.policy.optimizer_config.num_warmup_steps == 0,
    }
    failures = [label for label, passed in expected.items() if not passed]
    if failures:
        raise ValueError("Invalid algorithm-comparison configuration: " + ", ".join(failures))


def main():
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_experiment_config(cfg)
    initialize_ray(cfg)
    try:
        PolicyLossRegistry.register("signed_reinforce", signed_reinforce_policy_loss)
        AdvantageEstimatorRegistry.register("signed_reinforce", signed_outcome_advantage)
        AdvantageEstimatorRegistry.register(
            "batch_norm_reinforce", batch_normalized_outcome_advantage
        )
        validate_cfg(cfg)
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
