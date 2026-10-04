"""CPU native-Adam checks for compensated A-only switches and clipping.

Validates local algebra and observed-window moments, not RL performance,
counterfactual optimizer history, full-model FSDP, or useful accumulated rank.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch


def optimizer(a, b):
    return torch.optim.AdamW(
        [a, b], lr=1.5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=False
    )


def clip_gradients(a, b, gradient, scale):
    a.grad = scale * b.detach().T @ gradient
    b.grad = scale * gradient @ a.detach().T
    raw_a = a.grad.clone()
    raw_b = b.grad.clone()
    norm = torch.nn.utils.clip_grad_norm_([a, b], 1.0).item()
    coefficient = min(1.0, 1.0 / (norm + 1e-6))
    return raw_a, raw_b, coefficient


def run_case(seed, active_clip):
    generator = torch.Generator().manual_seed(seed)
    a = torch.nn.Parameter(torch.randn(1, 11, generator=generator, dtype=torch.float64) * 0.1)
    b = torch.nn.Parameter(torch.randn(7, 1, generator=generator, dtype=torch.float64) * 0.1)
    base = torch.randn(7, 11, generator=generator, dtype=torch.float64) * 0.1
    opt = optimizer(a, b)
    scale = 32
    window = []
    for step in range(1, 41):
        if step == 21:
            first = a.detach().T / a.detach().norm()
            extra = torch.randn(11, 1, generator=generator, dtype=torch.float64)
            extra -= first @ (first.T @ extra)
            q = torch.cat((first, extra / extra.norm()), dim=1)
        gradient = torch.randn(7, 11, generator=generator, dtype=torch.float64) * 0.01
        if step % 7 == 0:
            gradient *= 1000
        _, _, coefficient = clip_gradients(a, b, gradient, scale)
        if step >= 21:
            window.append(gradient * coefficient)
        opt.step()

    c = a.detach().norm() * torch.tensor([[math.cos(0.25)], [math.sin(0.25)]], dtype=torch.float64)
    new_a = (q @ c).T
    m = torch.zeros(7, 2, dtype=torch.float64)
    covariance = torch.zeros(7, 2, 2, dtype=torch.float64)
    direct_m = torch.zeros(7, 1, dtype=torch.float64)
    direct_v = torch.zeros_like(direct_m)
    for gradient in window:
        projected = gradient @ q
        m = 0.9 * m + 0.1 * projected
        covariance = 0.999 * covariance + 0.001 * projected[:, :, None] * projected[:, None, :]
        direct = scale * gradient @ new_a.T
        direct_m = 0.9 * direct_m + 0.1 * direct
        direct_v = 0.999 * direct_v + 0.001 * direct.square()
    mapped_m = scale * m @ c
    mapped_v = scale**2 * torch.einsum("i,oij,j->o", c[:, 0], covariance, c[:, 0])[:, None]
    moment_error = max(
        (mapped_m - direct_m).abs().max().item(), (mapped_v - direct_v).abs().max().item()
    )

    a_old, b_old = a.detach().clone(), b.detach().clone()
    ref_a = torch.nn.Parameter(a_old.clone())
    ref_b = torch.nn.Parameter(b_old.clone())
    ref_opt = optimizer(ref_a, ref_b)
    ref_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    switch_a = torch.nn.Parameter(new_a.clone())
    switch_b = torch.nn.Parameter(b_old.clone())
    switch_opt = optimizer(switch_a, switch_b)
    switch_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    switch_opt.state[switch_b]["exp_avg"].copy_(mapped_m)
    switch_opt.state[switch_b]["exp_avg_sq"].copy_(mapped_v)
    switch_opt.state[switch_b]["step"].fill_(20)
    compensated = base + scale * b_old @ (a_old - new_a)
    continuity_error = (
        (compensated + scale * switch_b @ switch_a - base - scale * b_old @ a_old)
        .abs()
        .max()
        .item()
    )
    fresh = torch.randn(7, 11, generator=generator, dtype=torch.float64) * (
        10 if active_clip else 0.01
    )
    raw_ref, _, clip_ref = clip_gradients(ref_a, ref_b, fresh, scale)
    raw_switch, _, clip_switch = clip_gradients(switch_a, switch_b, fresh, scale)
    raw_error = (raw_ref - raw_switch).abs().max().item()
    ref_opt.step()
    switch_opt.step()
    a_state_error = max(
        (ref_opt.state[ref_a][key] - switch_opt.state[switch_a][key]).abs().max().item()
        for key in ("exp_avg", "exp_avg_sq")
    )
    update_error = ((ref_a.detach() - a_old) - (switch_a.detach() - new_a)).abs().max().item()
    delta_b = switch_b.detach() - b_old
    u, v = b_old / b_old.norm(), a_old.T / a_old.norm()
    normal_b = delta_b - u @ (u.T @ delta_b)
    normal_a = new_a - (new_a @ v) @ v.T
    normal_first_order_update_norm = scale * normal_b.norm().item() * normal_a.norm().item()
    if max(moment_error, continuity_error, raw_error) > 1e-12:
        raise AssertionError("Observed-window moment or raw-gradient identity failed")
    if active_clip:
        if clip_ref == clip_switch or a_state_error <= 1e-12:
            raise AssertionError("Expected the active-clipping counterexample")
    elif a_state_error != 0 or update_error > 1e-12 or clip_ref != 1 or clip_switch != 1:
        raise AssertionError("Inactive-clipping A state continuity failed")
    counters = [switch_opt.state[param]["step"].item() for param in (switch_a, switch_b)]
    if counters != [41, 21] or normal_first_order_update_norm <= 0:
        raise AssertionError("Expected separate counters and nonzero normal update")
    return {
        "seed": seed,
        "fresh_gradient_clips": active_clip,
        "observed_window_moment_max_error": moment_error,
        "forward_compensation_max_error": continuity_error,
        "raw_A_gradient_max_error": raw_error,
        "reference_clip_scale": clip_ref,
        "switched_clip_scale": clip_switch,
        "A_moment_max_difference_after_fresh_update": a_state_error,
        "A_parameter_increment_max_difference": update_error,
        "switched_native_counters_A_B": counters,
        "normal_first_order_update_norm": normal_first_order_update_norm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cases = [run_case(seed, clip) for seed in (17, 29, 43, 71) for clip in (False, True)]
    report = {
        "passed": True,
        "cases": cases,
        "scope": "CPU float64 linear adapter, native AdamW, 40 observed updates and last-20 candidate moments; one fresh update. Local raw-gradient/state/window identities and active-clipping counterexamples only. No full-model BF16/FSDP, counterfactual native history, learned rank spectra, or RL-performance claim; running experiment unchanged.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(cases)} one-sided native-Adam cases; wrote {args.output}")


if __name__ == "__main__":
    main()
