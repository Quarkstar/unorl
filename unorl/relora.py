"""Restart scheduling, response-prefix probes and low-rank spectrum diagnostics."""

import math

import torch


def restart_multiplier(completed_updates, merge_interval, warmup_updates):
    """LR for the next update; initial training has no warmup.

    With interval 40 / warmup 5, updates 41..45 use 0, .25, .5, .75, 1.
    Native scheduler.step runs before the merge, preparing this next-update LR.
    """
    if merge_interval <= 0 or warmup_updates not in range(merge_interval):
        raise ValueError("Restart warmup must be nonnegative and shorter than the cycle")
    if warmup_updates == 1:
        raise ValueError("A restart ramp needs at least two updates, or zero to disable")
    if warmup_updates == 0 or completed_updates < merge_interval:
        return 1.0
    position = completed_updates % merge_interval
    return min(1.0, position / (warmup_updates - 1))


def trajectory_prefix(data, response_tokens=128):
    """Unpad one real math prompt and retain a bounded response prefix."""
    sequence = data["sequences"][0].detach()
    attention = data["attention_mask"][0].bool()
    response_mask = data["response_mask"][0].bool()
    prompt_end = sequence.numel() - response_mask.numel()
    prompt = sequence[:prompt_end][attention[:prompt_end]]
    response = sequence[prompt_end:][response_mask][:response_tokens]
    if not prompt.numel() or not response.numel():
        raise ValueError("Merge probe needs a valid prompt and response")
    return torch.cat([prompt, response]).cpu(), response.numel()


@torch.no_grad()
def response_log_distribution(model, tokens, response_length):
    """Only materialize response-prefix vocabulary logits, not full-trajectory logits."""
    device = next(model.parameters()).device
    tokens = tokens.to(device).unsqueeze(0)
    logits = model(tokens, logits_to_keep=response_length + 1, use_cache=False).logits[:, :-1]
    return logits.float().log_softmax(-1)


@torch.no_grad()
def distribution_shift(before, after, response_ids):
    """Teacher-forced old/new distribution changes on identical response tokens."""
    labels = response_ids.to(before.device).reshape(1, -1, 1)
    chosen_before = before.gather(-1, labels)
    chosen_after = after.gather(-1, labels)
    difference = chosen_after - chosen_before
    kl = (before.exp() * (before - after)).sum(-1)
    return {
        "relora/probe_response_tokens": float(labels.numel()),
        "relora/probe_response_kl_mean": kl.mean().clamp_min(0).item(),
        "relora/probe_chosen_logprob_mean_abs_diff": difference.abs().mean().item(),
        "relora/probe_chosen_logprob_max_abs_diff": difference.abs().max().item(),
        "relora/probe_argmax_flip_fraction": (before.argmax(-1) != after.argmax(-1))
        .float()
        .mean()
        .item(),
    }


@torch.no_grad()
def accumulated_spectrum(factors):
    """Singular values of sum(B @ A), via thin QR and a cycle-sized SVD."""
    if not factors:
        raise ValueError("At least one factor pair is required")
    b = torch.cat([pair[1].float() for pair in factors], dim=1)
    a_t = torch.cat([pair[0].float() for pair in factors], dim=0).T
    _, rb = torch.linalg.qr(b, mode="reduced")
    _, ra = torch.linalg.qr(a_t, mode="reduced")
    values = torch.linalg.svdvals(rb @ ra.T)
    energy = values.square().sum().item()
    leading = values[0].item()
    if not math.isfinite(energy):
        raise FloatingPointError("Nonfinite accumulated-update spectrum")
    return {
        "singular_values": values.tolist(),
        "stable_rank": energy / max(leading**2, 1e-30),
        "energy_outside_first_direction": max(0.0, 1 - leading**2 / max(energy, 1e-30))
        if energy
        else 0.0,
        "update_l2": math.sqrt(energy),
    }
