"""CPU native-Adam check of compensated rank-one sign charts; no RL training."""

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def check_case(seed, lr):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    scale, beta1, beta2, epsilon = 32, 0.9, 0.999, 1e-8
    a = torch.randn(1, 11, generator=generator, dtype=dtype) * 0.1
    b = torch.randn(7, 1, generator=generator, dtype=dtype) * 0.1
    base = torch.randn(7, 11, generator=generator, dtype=dtype) * 0.1
    ma, va, mb, vb = [torch.zeros_like(t) for t in (a, a, b, b)]
    # Observe fixed-coordinate gradients with native global clipping. This is
    # recorded-history symmetry, not an alternate training trajectory.
    parameters = [torch.nn.Parameter(a.clone()), torch.nn.Parameter(b.clone())]
    for _ in range(20):
        gradient = torch.randn(7, 11, generator=generator, dtype=dtype)
        parameters[0].grad = scale * b.T @ gradient
        parameters[1].grad = scale * gradient @ a.T
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        ga, gb = [p.grad for p in parameters]
        ma.lerp_(ga, 1 - beta1)
        mb.lerp_(gb, 1 - beta1)
        va.mul_(beta2).addcmul_(ga, ga, value=1 - beta2)
        vb.mul_(beta2).addcmul_(gb, gb, value=1 - beta2)
    fresh = torch.randn(7, 11, generator=generator, dtype=dtype)
    results = []
    for sign in (1, -1):
        aa, bb = torch.nn.Parameter(sign * a), torch.nn.Parameter(b.clone())
        optimizer = torch.optim.AdamW(
            [aa, bb], lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=0, foreach=False
        )
        for parameter, mean, variance in ((aa, ma, va), (bb, sign * mb, vb)):
            optimizer.state[parameter] = {
                "step": torch.tensor(20.0),
                "exp_avg": mean.clone(),
                "exp_avg_sq": variance.clone(),
            }
        chart_base = base + (1 - sign) * scale * (b @ a)
        before = chart_base + scale * bb @ aa
        aa.grad, bb.grad = scale * bb.detach().T @ fresh, scale * fresh @ aa.detach().T
        norm = torch.nn.utils.clip_grad_norm_([aa, bb], 1.0)
        optimizer.step()
        directions = []
        for parameter in (aa, bb):
            state = optimizer.state[parameter]
            assert state["step"] == 21
            directions.append(
                (state["exp_avg"] / (1 - beta1**21))
                / ((state["exp_avg_sq"] / (1 - beta2**21)).sqrt() + epsilon)
            )
        da, db = directions
        descent = scale * (b @ da + db @ (sign * a))
        after = chart_base + scale * bb.detach() @ aa.detach()
        results.append((before, after, descent, da, db, float(norm)))
    original, flipped = results
    continuity_error = (original[0] - flipped[0]).abs().max().item()
    descent_error = (original[2] - flipped[2]).abs().max().item()
    predicted_difference = -2 * scale * lr**2 * (original[4] @ original[3])
    actual_difference = flipped[1] - original[1]
    second_order_error = (actual_difference - predicted_difference).abs().max().item()
    assert original[5] == flipped[5]
    assert continuity_error < 1e-12 and descent_error < 1e-12
    assert second_order_error < 1e-12
    return {
        "seed": seed,
        "lr": lr,
        "continuity_max_abs_error": continuity_error,
        "first_order_descent_max_abs_error": descent_error,
        "second_order_formula_max_abs_error": second_order_error,
        "actual_one_step_weight_difference_l2": actual_difference.norm().item(),
        "predicted_second_order_difference_l2": predicted_difference.norm().item(),
        "native_counter_after": 21,
    }


def main():
    torch.set_num_threads(2)
    cases = [check_case(seed, lr) for seed in (17, 29, 43, 71) for lr in (1.5e-5, 3e-4)]
    report = {
        "method": "compensated A sign flip with exact signed first-moment mapping",
        "cases": cases,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "CPU float64 native AdamW with zero decay, 20 observed fixed-coordinate "
            "clipped gradients and one fresh clipped update. Demonstrates first-order "
            "descent invariance and the nonzero second-order difference. Not full-model "
            "BF16 continuity, identical long trajectories, rank growth or RL performance. "
            "No live experiment weights, optimizer states or settings were changed."
        ),
    }
    path = ROOT / "research/data/adapter-sign-chart-native-validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
