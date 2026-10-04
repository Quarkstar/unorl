"""CPU proof of prepared-coordinate moments on an observed training trajectory."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def diagnose():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(57)

    def random(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    scale = 4.0
    base = random(7, 11)
    a = torch.nn.Parameter(random(1, 11) * 0.1)
    b = torch.nn.Parameter(random(7, 1) * 0.1)
    next_a = random(1, 11) * 0.1
    next_b = random(7, 1) * 0.1
    current = torch.optim.AdamW([a, b], lr=0.003, weight_decay=0)
    projected = [
        torch.nn.Parameter(torch.zeros_like(next_a)),
        torch.nn.Parameter(torch.zeros_like(next_b)),
    ]
    oracle = [
        torch.nn.Parameter(torch.zeros_like(next_a)),
        torch.nn.Parameter(torch.zeros_like(next_b)),
    ]
    projected_optimizer = torch.optim.AdamW(projected, lr=0.003, weight_decay=0)
    oracle_optimizer = torch.optim.AdamW(oracle, lr=0.003, weight_decay=0)
    max_gradient_error = 0.0
    max_moment_error = 0.0
    clipping_multipliers = []
    for _ in range(12):
        current.zero_grad(set_to_none=True)
        accum_a = torch.zeros_like(next_a)
        accum_b = torch.zeros_like(next_b)
        dense_oracle = torch.zeros_like(base)
        # Unequal microbatch lengths, with normalization across all 10 tokens.
        for tokens in (2, 5, 3):
            x = random(tokens, 11)
            target = random(tokens, 7)
            y = F.linear(x, base) + scale * F.linear(F.linear(x, a), b)

            def observe(dy, x=x):
                nonlocal accum_a, accum_b, dense_oracle
                with torch.no_grad():
                    accum_a += scale * (dy @ next_b).T @ x
                    accum_b += scale * dy.T @ (x @ next_a.T)
                    # Only the independent small-fixture oracle constructs G.
                    dense_oracle += dy.T @ x

            y.register_hook(observe)
            ((y - target).square().sum() / (10 * 7)).backward()
        raw_norm = torch.nn.utils.clip_grad_norm_([a, b], max_norm=0.1)
        multiplier = min(1.0, 0.1 / (raw_norm.item() + 1e-6))
        clipping_multipliers.append(multiplier)
        grads = [accum_a * multiplier, accum_b * multiplier]
        reference = [
            scale * next_b.T @ dense_oracle * multiplier,
            scale * dense_oracle @ next_a.T * multiplier,
        ]
        for actual, expected, parameter, reference_parameter in zip(
            grads, reference, projected, oracle
        ):
            max_gradient_error = max(max_gradient_error, (actual - expected).abs().max().item())
            parameter.grad = actual
            reference_parameter.grad = expected
        projected_optimizer.step()
        oracle_optimizer.step()
        for parameter, reference_parameter in zip(projected, oracle):
            for key in ("exp_avg", "exp_avg_sq"):
                error = (
                    (
                        projected_optimizer.state[parameter][key]
                        - oracle_optimizer.state[reference_parameter][key]
                    )
                    .abs()
                    .max()
                    .item()
                )
                max_moment_error = max(max_moment_error, error)
        current.step()
    probe = random(5, 11)
    before = F.linear(probe, base) + scale * F.linear(F.linear(probe, a), b)
    with torch.no_grad():
        base += scale * (b @ a - next_b @ next_a)
        a.copy_(next_a)
        b.copy_(next_b)
    after = F.linear(probe, base) + scale * F.linear(F.linear(probe, a), b)
    forward_error = (before - after).abs().max().item()
    # Install observed-coordinate moments, rather than rescale old moments.
    for active, observer in zip((a, b), projected):
        current.state[active] = {
            key: value.clone() for key, value in projected_optimizer.state[observer].items()
        }
    assert max_gradient_error < 1e-12
    assert max_moment_error < 1e-12
    assert forward_error < 1e-12
    assert base.grad is None
    assert all(current.state[p]["step"].item() == 12 for p in (a, b))
    return {
        "observed_updates": 12,
        "microbatch_token_counts": [2, 5, 3],
        "dtype": "float64",
        "seed": 57,
        "native_optimizer": "torch.optim.AdamW",
        "max_projected_gradient_error_vs_dense_oracle": max_gradient_error,
        "max_native_moment_error_vs_dense_oracle": max_moment_error,
        "max_boundary_forward_error": forward_error,
        "clipping_multiplier_range": [min(clipping_multipliers), max(clipping_multipliers)],
        "installed_observer_step_counters": [current.state[p]["step"].item() for p in (a, b)],
        "observed_base_has_no_gradient": base.grad is None,
        "limitations": "One CPU linear-layer fixture with fixed prepared factors. Dense G exists only in the independent oracle. Proves observed-coordinate projection, aggregated/clipped moment bookkeeping and real-arithmetic compensation at double precision; not counterfactual policy history, FSDP, mixed precision, production hook lifetime, peak memory, rank improvement or accuracy.",
    }


def main():
    result = diagnose()
    output = ROOT / "research/data/prepared-adapter-moment-validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
