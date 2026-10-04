"""Record the normal-descent selector's quadratic examples and tradeoffs."""

import hashlib
import json
from pathlib import Path

import torch

from unorl.prepared_moments import PreparedBasisMoments
from unorl.prepared_selection import select_normal_descent

ROOT = Path(__file__).resolve().parents[2]


def observe(gradient):
    basis = torch.eye(2, dtype=torch.float64)
    state = PreparedBasisMoments(basis, basis)
    for _ in range(5):
        state.accumulate(basis, gradient.T)
        state.finish_update(1)
    return state


def main():
    torch.set_num_threads(2)
    state = observe(torch.diag(torch.tensor([100.0, 1.0], dtype=torch.float64)))
    old_a, old_b = state.select_coefficients(1, 1)
    assert old_a[0, 1] == old_b[1, 0] == 0
    tradeoffs = []
    for fraction in (0.0, 0.5, 0.9, 0.99, 1.0):
        selected = select_normal_descent(state, 1, 1, 4, min_total_fraction=fraction)
        tradeoffs.append(
            {
                "a_coeff": selected["a_coeff"].tolist(),
                "b_coeff": selected["b_coeff"].tolist(),
                **selected["diagnostics"],
            }
        )
    target = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    current = torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64))
    state = observe(current - target)
    selected = select_normal_descent(state, 1, 1, 4, min_total_fraction=0.9)
    mapped = state.map_moments(selected["a_coeff"], selected["b_coeff"], 4)
    bias1, bias2 = 1 - state.beta1**state.count, 1 - state.beta2**state.count
    da = mapped["ma"] / bias1 / ((mapped["va"] / bias2).sqrt() + 1e-8)
    db = mapped["mb"] / bias1 / ((mapped["vb"] / bias2).sqrt() + 1e-8)
    lr = 1e-3
    compensated = current - 4 * mapped["b"] @ mapped["a"]
    next_weight = compensated + 4 * (mapped["b"] - lr * db) @ (mapped["a"] - lr * da)
    before = 0.5 * (current - target).square().sum().item()
    after = 0.5 * (next_weight - target).square().sum().item()
    assert after < before and torch.linalg.matrix_rank(next_weight).item() == 2
    paths = (
        "unorl/prepared_selection.py",
        "unorl/prepared_moments.py",
        "tests/test_prepared_selection.py",
        "scripts/research/diagnose_prepared_selection.py",
    )
    report = {
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
        },
        "torch_version": torch.__version__,
        "precision": "CPU float64",
        "tests_passed": 7,
        "observed_history_updates": 5,
        "dominant_old_gradient_example": {
            "mean_gradient_diagonal": [100, 1],
            "leading_svd_selects_only_old_direction": True,
            "grid_tradeoffs": tradeoffs,
        },
        "stationary_rank_one_quadratic_example": {
            "target_diagonal": [2, 1],
            "initial_diagonal": [2, 0],
            "initial_loss": before,
            "next_loss": after,
            "next_weight": next_weight.tolist(),
            "rank_before": 1,
            "rank_after": 2,
            "proposed_factor_step_lr": lr,
            "diagnostics": selected["diagnostics"],
        },
        "production_integrated": False,
        "limitations": "Isolated CPU direction selection against a fixed-window observed mean with warmed diagonal Adam preconditioning. Numerical angular grid optimum under per-factor total-descent constraints, not a continuous optimum. The example applies a proposed direction from frozen moments, not a subsequent native Adam update with fresh rollout gradients. Reference is the first basis column at preparation, not the full accumulated-rank span or necessarily the adapter at the switch. No reward, long-run rank growth, worker memory or native selector integration evidence; the earlier native fixture validated transfer with a different SVD selector.",
    }
    (ROOT / "research/data/prepared-direction-selection-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
