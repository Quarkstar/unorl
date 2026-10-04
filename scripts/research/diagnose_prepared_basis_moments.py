"""CPU proof of adaptive direction selection from recorded projected moments."""

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def diagnose():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(73)
    dtype = torch.float64
    # Bases stay fixed throughout preparation; coefficients are selected later.
    qout = torch.linalg.qr(torch.randn(13, 2, generator=generator, dtype=dtype)).Q
    qin = torch.linalg.qr(torch.randn(11, 2, generator=generator, dtype=dtype)).Q
    mb = torch.zeros(13, 2, dtype=dtype)
    ma = torch.zeros(2, 11, dtype=dtype)
    cb = torch.zeros(13, 2, 2, dtype=dtype)
    ca = torch.zeros(11, 2, 2, dtype=dtype)
    gradients = []
    scale = 4.0
    beta1, beta2 = 0.9, 0.999
    for _ in range(16):
        gb = torch.zeros_like(mb)
        ga = torch.zeros_like(ma)
        oracle = torch.zeros(13, 11, dtype=dtype)
        for tokens in (2, 5, 3):
            x = torch.randn(tokens, 11, generator=generator, dtype=dtype)
            dy = torch.randn(tokens, 13, generator=generator, dtype=dtype) / 10
            gb += dy.T @ (x @ qin)
            ga += (dy @ qout).T @ x
            # Dense weight gradient exists only in this small reference oracle.
            oracle += dy.T @ x
        clip = 0.1 + 0.5 * torch.rand((), generator=generator).item()
        gb *= clip
        ga *= clip
        gradients.append(oracle * clip)
        mb = beta1 * mb + (1 - beta1) * gb
        ma = beta1 * ma + (1 - beta1) * ga
        cb = beta2 * cb + (1 - beta2) * torch.einsum("oi,oj->oij", gb, gb)
        ca = beta2 * ca + (1 - beta2) * torch.einsum("io,jo->oij", ga, ga)
    core = qout.T @ mb
    assert torch.allclose(core, ma @ qin, atol=1e-12, rtol=0)
    left, singular, right = torch.linalg.svd(core)
    b_coeff = left[:, :1] * 0.3
    a_coeff = right[:1] * 0.7
    b = qout @ b_coeff
    a = a_coeff @ qin.T
    projected = [torch.nn.Parameter(torch.zeros_like(a)), torch.nn.Parameter(torch.zeros_like(b))]
    optimizer = torch.optim.AdamW(projected, weight_decay=0)
    for gradient in gradients:
        projected[0].grad = scale * b.T @ gradient
        projected[1].grad = scale * gradient @ a.T
        optimizer.step()
    transported_ma = scale * b_coeff.T @ ma
    transported_mb = scale * mb @ a_coeff.T
    transported_va = scale**2 * torch.einsum(
        "i,oij,j->o", b_coeff[:, 0], ca, b_coeff[:, 0]
    ).reshape(1, -1)
    transported_vb = scale**2 * torch.einsum("i,oij,j->o", a_coeff[0], cb, a_coeff[0]).reshape(
        -1, 1
    )
    errors = []
    for p, m, v in zip(
        projected, (transported_ma, transported_mb), (transported_va, transported_vb)
    ):
        errors.extend(
            [
                (optimizer.state[p]["exp_avg"] - m).abs().max().item(),
                (optimizer.state[p]["exp_avg_sq"] - v).abs().max().item(),
            ]
        )
        assert optimizer.state[p]["step"].item() == 16
    assert max(errors) < 1e-12
    # A strict rank-one optimum can have a nonzero normal gradient.
    target = torch.diag(torch.tensor([2.0, 1.0], dtype=dtype))
    old_a = torch.tensor([[1.0, 0.0]], dtype=dtype)
    old_b = torch.tensor([[2.0], [0.0]], dtype=dtype)
    old_weight = old_b @ old_a
    gradient = old_weight - target
    assert torch.count_nonzero(old_b.T @ gradient) == 0
    assert torch.count_nonzero(gradient @ old_a.T) == 0
    old_loss = 0.5 * (old_weight - target).square().sum().item()
    residual_a = torch.tensor([[0.0, 1.0]], dtype=dtype)
    residual_b = torch.tensor([[0.0], [1.0]], dtype=dtype)
    total = old_weight + residual_b @ residual_a
    assert torch.linalg.matrix_rank(total).item() == 2
    assert torch.equal(total, target)
    return {
        "observed_updates": 16,
        "candidate_basis_width": 2,
        "coefficients_selected_after_history": True,
        "max_native_adam_moment_error": max(errors),
        "projected_mean_gradient_core_top_singular_value": singular[0].item(),
        "rank_one_stationary_example": {
            "rank_one_loss": old_loss,
            "rank_two_loss": 0.0,
            "old_adapter_gradient_zero": True,
            "normal_weight_gradient_nonzero": True,
        },
        "limitations": "CPU double-precision identities with fixed per-cycle bases and observed clipping. Dense gradients retained only by the oracle. Adaptive coefficients are constant when replaying observed gradients, not a counterfactual policy trajectory. Rank-two matrix example illustrates capacity, not RL performance. No FSDP, GPU memory or production collector validation.",
    }


def main():
    report = diagnose()
    (ROOT / "research/data/prepared-basis-moment-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
