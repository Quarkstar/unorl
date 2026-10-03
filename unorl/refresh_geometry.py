"""Low-rank diagnostics of an adapter's first-order update space."""

import math

import torch


@torch.no_grad()
def update_space_residual(a, b, factors):
    """Measure the unavailable part of sum(column @ row), without dense weights.

    Each factor is (row [1, in], column [out, 1]), as in update_factors.
    The residual is (I - P_b) D (I - P_a). This describes the closest
    first-order adapter update, not exact finite-step Adam transport.
    Zero factors are supported: their corresponding projector is zero.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != 1 or b.shape[1] != 1:
        raise ValueError("Expected rank-one adapter factors")
    a, b = a.double(), b.double()
    aa, bb = a.square().sum(), b.square().sum()
    target, residual = [], []
    for row, column in factors:
        if row.shape != a.shape or column.shape != b.shape:
            raise ValueError("Update factors must match the adapter dimensions")
        row, column = row.double(), column.double()
        target.append((row, column))
        perpendicular_row = row - (row @ a.T) * a / aa if aa > 0 else row
        perpendicular_column = column - b * (b.T @ column) / bb if bb > 0 else column
        residual.append((perpendicular_row, perpendicular_column))

    def energy(terms):
        return sum(
            float((left_row * right_row).sum() * (left_column * right_column).sum())
            for left_row, left_column in terms
            for right_row, right_column in terms
        )

    total, unavailable = energy(target), energy(residual)
    if not math.isfinite(total) or not math.isfinite(unavailable):
        raise FloatingPointError("Nonfinite update-space energy")
    total, unavailable = max(0.0, total), max(0.0, unavailable)
    return {
        "target_l2": math.sqrt(total),
        "unavailable_l2": math.sqrt(unavailable),
        "relative_residual": math.sqrt(unavailable / total) if total > 0 else 0.0,
    }
