"""Select prepared rank-one factors by predicted normal-gradient descent.

The reference directions are the first columns of the fixed candidate bases.
The score concerns their rank-one tangent space, not every historical adapter
direction. It predicts a first-order step against the observed mean gradient;
it is neither a reward guarantee nor a counterfactual Adam trajectory.
"""

import math

import torch


@torch.no_grad()
def select_normal_descent(
    observer, a_norm, b_norm, scale, epsilon=1e-8, angles=256, chunk=16, min_total_fraction=0.0
):
    """Maximize observed normal descent over two independent angular grids.

    Adam's proposed factor directions depend on the opposite factor only,
    so the normal-descent objective separates into two one-dimensional
    searches. Each side can be constrained to retain min_total_fraction of
    its best total-descent score; their sum then satisfies the same bound.
    No dense weight gradient or angle-pair grid is constructed.
    The returned diagnostic must be checked before treating a switch as
    supported: a maximum of zero is valid evidence of no observed benefit.
    """
    if observer.pending_microbatches or not observer.count:
        raise ValueError("Selection requires a completed nonempty history")
    if any(not math.isfinite(x) or x <= 0 for x in (a_norm, b_norm, scale, epsilon)):
        raise ValueError("Factor norms, scale and epsilon must be finite and positive")
    if type(angles) is not int or angles < 4 or angles % 4:
        raise ValueError("Angular grid must be a positive multiple of four, at least four")
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("Candidate chunk must be a positive integer")
    if not math.isfinite(min_total_fraction) or not 0 <= min_total_fraction <= 1:
        raise ValueError("Minimum total-descent fraction must be in [0,1]")

    bias1 = 1 - observer.beta1**observer.count
    bias2 = 1 - observer.beta2**observer.count
    theta = torch.arange(angles, device=observer.qin.device, dtype=observer.qin.dtype)
    theta *= math.pi / angles
    coefficients = torch.stack((theta.cos(), theta.sin()), dim=1)
    # Exact axes, rather than small trigonometric residuals at pi/2.
    coefficients[0] = coefficients.new_tensor([1, 0])
    coefficients[angles // 2] = coefficients.new_tensor([0, 1])

    def search(mean, packed, reference, factor_norm):
        mean = mean / bias1
        # Row/column of the mean gradient outside the reference input/output
        # direction. qin/qout have orthonormal columns by observer validation.
        fresh = mean[:, 1]
        normal = fresh - reference * torch.dot(reference, fresh)
        all_normal = mean.new_empty(angles)
        all_total = mean.new_empty(angles)
        for start in range(0, angles, chunk):
            c = coefficients[start : start + chunk] * factor_norm
            gradient = scale * (mean @ c.T)
            terms = torch.stack(
                (
                    packed[:, :1] * c[:, 0].square(),
                    2 * packed[:, 1:2] * c[:, 0] * c[:, 1],
                    packed[:, 2:] * c[:, 1].square(),
                )
            )
            variance = terms.sum(dim=0)
            tolerance = 32 * torch.finfo(variance.dtype).eps * terms.abs().sum(dim=0)
            if (variance < -tolerance).any():
                raise FloatingPointError("Prepared cross moments yield negative variance")
            variance = variance.clamp_min(0) * (scale**2 / bias2)
            direction = gradient / (variance.sqrt() + epsilon)
            normal_gradient = scale * normal[:, None] * c[:, 1]
            scores = (normal_gradient * direction).sum(dim=0)
            total = (gradient * direction).sum(dim=0)
            if not torch.isfinite(scores).all() or not torch.isfinite(total).all():
                raise FloatingPointError("Nonfinite prepared descent score")
            all_normal[start : start + len(c)] = scores
            all_total[start : start + len(c)] = total
        total_max = all_total.max().item()
        feasible = all_total >= min_total_fraction * total_max
        index = int(all_normal.masked_fill(~feasible, -math.inf).argmax())
        best = (coefficients[index] * factor_norm, all_total[index].item(), index)
        return best, all_normal[index].item(), total_max

    # Selecting B controls the A update; selecting A controls the B update.
    selected_b, gain_a, max_a = search(observer.ma.T, observer.ca.T, observer.qin[:, 0], b_norm)
    selected_a, gain_b, max_b = search(observer.mb, observer.cb, observer.qout[:, 0], a_norm)
    total_gain = selected_a[1] + selected_b[1]
    normal_gain = gain_a + gain_b
    return {
        "a_coeff": selected_a[0].reshape(1, 2),
        "b_coeff": selected_b[0].reshape(2, 1),
        "diagnostics": {
            "predicted_normal_descent_per_lr": normal_gain,
            "predicted_total_descent_per_lr": total_gain,
            "maximum_total_descent_per_lr_in_grid": max_a + max_b,
            "total_descent_retained_fraction": total_gain / (max_a + max_b)
            if max_a + max_b
            else 0.0,
            "has_positive_normal_descent": normal_gain > 0,
            "a_grid_index": selected_a[2],
            "b_grid_index": selected_b[2],
            "angles_per_factor": angles,
            "minimum_total_descent_fraction_per_factor": min_total_fraction,
            "reference": "first columns of fixed per-window qin/qout",
            "scope": "first-order observed-mean loss prediction, no reward or cumulative-rank guarantee",
        },
    }
