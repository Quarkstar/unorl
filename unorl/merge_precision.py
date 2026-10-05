"""Read-only numerical measurements for a single LoRA merge boundary."""

import torch


def adam_step_counters(states):
    """Ignore frozen-parameter placeholders in native FSDP optimizer state."""
    counters = []
    for state in states:
        if not state:
            continue
        if not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
            raise ValueError("Nonempty native Adam state is missing counters or moments")
        counters.append(float(state["step"]))
    return counters


@torch.no_grad()
def weight_rounding_metrics(base, a, b, scale):
    """Separate native FP32 addition error from the subsequent BF16 cast.

    The reference uses FP64 factor multiplication. These are weight-space
    measurements; their magnitude alone does not establish a policy change.
    """
    if base.dtype != torch.float32:
        raise ValueError("The native training backbone must use FP32 storage")
    reference_delta = (b.double() @ a.double()) * scale
    reference = base.double() + reference_delta
    native = base + (b.float() @ a.float()) * scale
    receiver = native.to(torch.bfloat16).double()
    changed = reference_delta != 0
    return {
        "base_energy": base.double().square().sum().item(),
        "adapter_energy": reference_delta.square().sum().item(),
        "fp32_merge_error_energy": (native.double() - reference).square().sum().item(),
        "bf16_total_error_energy": (receiver - reference).square().sum().item(),
        "bf16_cast_error_energy": (receiver - native.double()).square().sum().item(),
        "changed_elements": changed.sum().item(),
        "changed_elements_lost_in_bf16": ((receiver == base.double()) & changed).sum().item(),
    }


@torch.no_grad()
def output_shift(before, after, response_ids):
    """Compare fixed-token logits; every response token has equal weight."""
    if before.shape != after.shape or before.shape[-2] != response_ids.numel():
        raise ValueError("Both forwards must use the same response positions")
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise FloatingPointError("Nonfinite boundary logits")
    before_logp = before.float().log_softmax(-1)
    after_logp = after.float().log_softmax(-1)
    labels = response_ids.to(before.device).reshape(1, -1, 1)
    chosen = after_logp.gather(-1, labels) - before_logp.gather(-1, labels)
    logits_delta = after.float() - before.float()
    kl = (before_logp.exp() * (before_logp - after_logp)).sum(-1)
    return {
        "response_tokens": response_ids.numel(),
        "kl_mean": kl.mean().clamp_min(0).item(),
        "chosen_logprob_mean_abs_shift": chosen.abs().mean().item(),
        "chosen_logprob_max_abs_shift": chosen.abs().max().item(),
        "argmax_flip_fraction": (before.argmax(-1) != after.argmax(-1)).float().mean().item(),
        "logit_mean_abs_shift": logits_delta.abs().mean().item(),
        "logit_rms_shift": logits_delta.square().mean().sqrt().item(),
        "logit_max_abs_shift": logits_delta.abs().max().item(),
    }


def aggregate_output_shifts(rows):
    """Weight means by tokens and take maxima across all ranks/examples."""
    tokens = sum(row["response_tokens"] for row in rows)
    if not tokens:
        raise ValueError("A precision probe needs response tokens")
    result = {"response_tokens": tokens}
    for key in rows[0]:
        if key == "response_tokens":
            continue
        if "max_abs" in key:
            result[key] = max(row[key] for row in rows)
        elif key == "logit_rms_shift":
            result[key] = (
                sum(row[key] ** 2 * row["response_tokens"] for row in rows) / tokens
            ) ** 0.5
        else:
            result[key] = sum(row[key] * row["response_tokens"] for row in rows) / tokens
    return result
