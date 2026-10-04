"""Independent dense oracles for the isolated prepared direction selector."""

import pytest
import torch

from unorl.prepared_moments import PreparedBasisMoments
from unorl.prepared_selection import select_normal_descent


def observe(gradients, qin=None, qout=None):
    gradient = gradients[0]
    qin = torch.eye(gradient.shape[1], dtype=gradient.dtype)[:, :2] if qin is None else qin
    qout = torch.eye(gradient.shape[0], dtype=gradient.dtype)[:, :2] if qout is None else qout
    state = PreparedBasisMoments(qin, qout, chunk_tokens=3)
    mean = torch.zeros_like(gradient)
    for gradient in gradients:
        state.accumulate(torch.eye(gradient.shape[1], dtype=gradient.dtype), gradient.T)
        state.finish_update(1)
        mean = state.beta1 * mean + (1 - state.beta1) * gradient
    return state, mean / (1 - state.beta1**state.count)


def proposed_step(state, selected, scale=4, epsilon=1e-8):
    mapped = state.map_moments(selected["a_coeff"], selected["b_coeff"], scale)
    bias1 = 1 - state.beta1**state.count
    bias2 = 1 - state.beta2**state.count
    da = mapped["ma"] / bias1 / ((mapped["va"] / bias2).sqrt() + epsilon)
    db = mapped["mb"] / bias1 / ((mapped["vb"] / bias2).sqrt() + epsilon)
    return mapped, da, db, scale * (mapped["b"] @ da + db @ mapped["a"])


def test_selector_escapes_dominant_old_svd_direction():
    gradient = torch.diag(torch.tensor([100.0, 1.0], dtype=torch.float64))
    state, mean = observe([gradient] * 4)
    old_a, old_b = state.select_coefficients(1, 1)
    assert old_a[0, 1] == old_b[1, 0] == 0
    selected = select_normal_descent(state, 1, 1, 4)
    assert torch.equal(selected["a_coeff"], torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    assert torch.equal(selected["b_coeff"], torch.tensor([[0.0], [1.0]], dtype=torch.float64))
    assert selected["diagnostics"]["has_positive_normal_descent"]
    # The cost of prioritizing the weak normal gradient must remain visible.
    assert selected["diagnostics"]["total_descent_retained_fraction"] < 0.02
    _, _, _, step = proposed_step(state, selected)
    assert (mean * step).sum() > 0


def test_normal_and_total_scores_match_independent_dense_weight_oracle():
    generator = torch.Generator().manual_seed(941)
    qin = torch.linalg.qr(torch.randn(11, 2, dtype=torch.float64, generator=generator)).Q
    qout = torch.linalg.qr(torch.randn(13, 2, dtype=torch.float64, generator=generator)).Q
    gradients = [torch.randn(13, 11, dtype=torch.float64, generator=generator) for _ in range(7)]
    state, mean = observe(gradients, qin, qout)
    selected = select_normal_descent(state, 0.7, 0.3, 4, angles=64, chunk=7)
    _, _, _, step = proposed_step(state, selected)
    left = torch.eye(13, dtype=torch.float64) - qout[:, :1] @ qout[:, :1].T
    right = torch.eye(11, dtype=torch.float64) - qin[:, :1] @ qin[:, :1].T
    normal = left @ mean @ right
    diagnostics = selected["diagnostics"]
    assert diagnostics["has_positive_normal_descent"]
    assert diagnostics["predicted_normal_descent_per_lr"] == pytest.approx(
        (normal * step).sum().item(), rel=1e-12, abs=1e-12
    )
    assert diagnostics["predicted_total_descent_per_lr"] == pytest.approx(
        (mean * step).sum().item(), rel=1e-12, abs=1e-12
    )
    other = select_normal_descent(state, 0.7, 0.3, 4, angles=64, chunk=1)
    assert torch.equal(other["a_coeff"], selected["a_coeff"])
    assert torch.equal(other["b_coeff"], selected["b_coeff"])


def test_no_normal_signal_is_reported_without_forcing_a_novel_switch():
    state, _ = observe([torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))])
    selected = select_normal_descent(state, 1, 1, 4)
    diagnostics = selected["diagnostics"]
    assert not diagnostics["has_positive_normal_descent"]
    assert diagnostics["predicted_normal_descent_per_lr"] == 0
    assert selected["a_coeff"][0, 1] == selected["b_coeff"][1, 0] == 0


def test_total_descent_constraint_limits_the_sacrifice_of_old_signal():
    state, _ = observe([torch.diag(torch.tensor([100.0, 1.0], dtype=torch.float64))] * 4)
    selected = select_normal_descent(state, 1, 1, 4, min_total_fraction=0.9)
    diagnostics = selected["diagnostics"]
    assert diagnostics["has_positive_normal_descent"]
    assert diagnostics["total_descent_retained_fraction"] >= 0.9
    assert diagnostics["predicted_normal_descent_per_lr"] > 0
    assert selected["a_coeff"][0, 1] > 0 and selected["b_coeff"][1, 0] > 0
    assert selected["a_coeff"][0, 0] > 0 and selected["b_coeff"][0, 0] > 0


def test_constrained_grid_optimum_matches_independent_enumeration():
    generator = torch.Generator().manual_seed(945)
    gradients = [torch.randn(4, 3, dtype=torch.float64, generator=generator) for _ in range(3)]
    state, mean = observe(gradients)
    normal = mean.clone()
    normal[0] = 0
    normal[:, 0] = 0
    angles, fraction = 16, 0.9
    coefficients = torch.stack(
        (
            (torch.arange(angles, dtype=torch.float64) * torch.pi / angles).cos(),
            (torch.arange(angles, dtype=torch.float64) * torch.pi / angles).sin(),
        ),
        dim=1,
    )
    coefficients[0] = torch.tensor([1.0, 0.0])
    coefficients[angles // 2] = torch.tensor([0.0, 1.0])
    old_a, old_b = coefficients[0].reshape(1, 2), coefficients[0].reshape(2, 1)
    total_a, total_b, normal_a, normal_b = [], [], [], []
    for c in coefficients:
        selected = {"a_coeff": c.reshape(1, 2), "b_coeff": old_b}
        mapped, _, db, _ = proposed_step(state, selected)
        step_b = 4 * db @ mapped["a"]
        total_b.append((mean * step_b).sum().item())
        normal_b.append((normal * step_b).sum().item())
        selected = {"a_coeff": old_a, "b_coeff": c.reshape(2, 1)}
        mapped, da, _, _ = proposed_step(state, selected)
        step_a = 4 * mapped["b"] @ da
        total_a.append((mean * step_a).sum().item())
        normal_a.append((normal * step_a).sum().item())
    allowed_a = [i for i, score in enumerate(total_b) if score >= fraction * max(total_b)]
    allowed_b = [i for i, score in enumerate(total_a) if score >= fraction * max(total_a)]
    expected = max(
        ((i, j) for i in allowed_a for j in allowed_b),
        key=lambda pair: normal_b[pair[0]] + normal_a[pair[1]],
    )
    selected = select_normal_descent(state, 1, 1, 4, angles=angles, min_total_fraction=fraction)
    diagnostics = selected["diagnostics"]
    assert (diagnostics["a_grid_index"], diagnostics["b_grid_index"]) == expected


def test_normal_selection_adds_rank_at_a_rank_one_stationary_point():
    target = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    current = torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64))
    gradient = current - target
    state, _ = observe([gradient] * 5)
    selected = select_normal_descent(state, 1, 1, 4)
    mapped, da, db, linear_step = proposed_step(state, selected)
    assert gradient[0].count_nonzero() == gradient[:, 0].count_nonzero() == 0
    # Compensate first, then apply the predicted factor step. This is a
    # deterministic quadratic matrix example, not a native RL optimizer run.
    base = current - 4 * mapped["b"] @ mapped["a"]
    lr = 1e-3
    next_weight = base + 4 * (mapped["b"] - lr * db) @ (mapped["a"] - lr * da)
    assert torch.linalg.matrix_rank(next_weight).item() == 2
    assert (next_weight - target).square().sum() < (current - target).square().sum()
    initial_loss = 0.5 * (current - target).square().sum()
    next_loss = 0.5 * (next_weight - target).square().sum()
    assert ((initial_loss - next_loss) / lr).item() == pytest.approx(
        (gradient * linear_step).sum().item(), rel=0.01
    )


def test_selector_rejects_incomplete_history_and_invalid_search():
    state, _ = observe([torch.eye(2, dtype=torch.float64)])
    with pytest.raises(ValueError, match="multiple of four"):
        select_normal_descent(state, 1, 1, 4, angles=5)
    with pytest.raises(ValueError, match="positive"):
        select_normal_descent(state, 1, 1, 4, epsilon=0)
    with pytest.raises(ValueError, match="fraction"):
        select_normal_descent(state, 1, 1, 4, min_total_fraction=1.1)
    state.accumulate(torch.eye(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="completed"):
        select_normal_descent(state, 1, 1, 4)
