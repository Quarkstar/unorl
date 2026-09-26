"""Observe which on-policy trajectories actually contribute to the policy update."""

import math
import statistics


def rollout_diagnostics(output, normalize_rewards):
    """Measure truncation and generator loss-mask selection without changing a batch.

    Advantage statistics use the same binary outcome transform as REINFORCE,
    before token-mean reduction and inference importance correction.
    """
    responses = output["response_ids"]
    masks = output["loss_masks"]
    rewards = output["rewards"]
    reasons = output["stop_reasons"]
    count = len(responses)
    if not count or not (len(masks) == len(rewards) == len(reasons) == count):
        raise ValueError("Inconsistent or empty rollout batch")
    if any(len(response) != len(mask) for response, mask in zip(responses, masks)):
        raise ValueError("Response/loss-mask length mismatch")
    scores = [float(sum(r) if isinstance(r, (list, tuple)) else r) for r in rewards]
    if any(score not in (-1.0, 1.0) for score in scores):
        raise ValueError("Diagnostics expect signed binary rewards")
    mean = statistics.mean(scores)
    std = statistics.pstdev(scores)
    advantages = (
        [(score - mean) / max(std, 1e-8) for score in scores] if normalize_rewards else scores
    )
    lengths = [len(response) for response in responses]
    active_tokens = [sum(mask) for mask in masks]
    active = [tokens > 0 for tokens in active_tokens]
    truncated = [reason == "length" for reason in reasons]
    completed = [reason == "stop" for reason in reasons]
    positive_mass = sum(max(a, 0) * n for a, n in zip(advantages, active_tokens))
    negative_mass = sum(max(-a, 0) * n for a, n in zip(advantages, active_tokens))
    truncated_negative_mass = sum(
        max(-a, 0) * n for a, n, trunc in zip(advantages, active_tokens, truncated) if trunc
    )
    metrics = {
        "rollout/truncated_fraction": sum(truncated) / count,
        "rollout/non_stop_fraction": sum(not c for c in completed) / count,
        "rollout/trainable_response_fraction": sum(active) / count,
        "rollout/trainable_token_fraction": sum(active_tokens) / max(sum(lengths), 1),
        "rollout/masked_incorrect_fraction": sum(
            score < 0 and not keep for score, keep in zip(scores, active)
        )
        / max(sum(score < 0 for score in scores), 1),
        "advantage/outcome_std": std,
        "advantage/active_response_mean": sum(a for a, keep in zip(advantages, active) if keep)
        / max(sum(active), 1),
        "advantage/completed_response_mean": sum(
            a for a, keep in zip(advantages, completed) if keep
        )
        / max(sum(completed), 1),
        "advantage/token_weighted_mean": sum(a * n for a, n in zip(advantages, active_tokens))
        / max(sum(active_tokens), 1),
        "advantage/positive_token_mass_fraction": positive_mass
        / max(positive_mass + negative_mass, 1e-8),
        "advantage/truncated_negative_mass_fraction": truncated_negative_mass
        / max(negative_mass, 1e-8),
    }
    for label, subset in [("completed", completed), ("truncated", truncated)]:
        if any(subset):
            metrics[f"rollout/{label}_correctness"] = sum(
                score > 0 for score, keep in zip(scores, subset) if keep
            ) / sum(subset)
            metrics[f"rollout/{label}_tokens_mean"] = statistics.mean(
                length for length, keep in zip(lengths, subset) if keep
            )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite rollout diagnostics")
    return metrics
