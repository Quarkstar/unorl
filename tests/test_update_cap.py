import pytest
import torch

from unorl.update_cap import cap_multiplier, interpolate_factors, update_norm


@pytest.mark.parametrize("zero_b", [False, True])
def test_cap_matches_dense_update_and_keeps_combined_norm_bounded(zero_b):
    torch.manual_seed(42)
    before, after = {}, {}
    for name, shape in [("one", (7, 5)), ("two", (4, 3))]:
        out_dim, in_dim = shape
        a, b = (
            torch.randn(1, in_dim, dtype=torch.float64),
            torch.randn(out_dim, 1, dtype=torch.float64),
        )
        if zero_b:
            b.zero_()
        before[name] = a, b, 32.0
        after[name] = a + torch.randn_like(a), b + torch.randn_like(b), 32.0
    dense = (
        sum(
            ((b @ a - before[name][1] @ before[name][0]) * scale).square().sum()
            for name, (a, b, scale) in after.items()
        )
        .sqrt()
        .item()
    )
    assert update_norm(before, after) == pytest.approx(dense)
    budget = dense / 4
    multiplier, diagnostics = cap_multiplier(before, after, budget)
    assert 0 < multiplier < 1
    applied = interpolate_factors(before, after, multiplier)
    assert update_norm(before, applied) <= budget * (1 + 1e-12)
    assert diagnostics["conservative_bound_l2"] == pytest.approx(budget)
    for name in before:
        for index in [0, 1]:
            torch.testing.assert_close(
                applied[name][index] - before[name][index],
                multiplier * (after[name][index] - before[name][index]),
            )


def test_under_budget_native_update_is_unchanged():
    before = {"one": (torch.ones(1, 3), torch.zeros(4, 1), 32)}
    after = {"one": (torch.ones(1, 3), torch.ones(4, 1) * 0.001, 32)}
    multiplier, _ = cap_multiplier(before, after, 1)
    assert multiplier == 1
    assert update_norm(before, after) < 1


def test_quadratic_cross_term_cannot_be_ignored():
    before = {
        "one": (torch.zeros(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64), 1)
    }
    after = {
        "one": (torch.ones(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64), 1)
    }
    multiplier, _ = cap_multiplier(before, after, 0.25)
    assert multiplier == pytest.approx(0.5)
    assert update_norm(before, interpolate_factors(before, after, multiplier)) == pytest.approx(
        0.25
    )


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="budget"):
        cap_multiplier({}, {}, budget)
