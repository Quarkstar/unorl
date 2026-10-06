"""Bound effective LoRA weight updates without forming dense matrices."""

import math

from unorl.relora_refresh import factor_inner, update_factors


def update_norm(before, after):
    energy = 0.0
    for name, (a, b, scale) in after.items():
        old_a, old_b, old_scale = before[name]
        if scale != old_scale:
            raise ValueError("Adapter scale changed during the optimizer update")
        terms = update_factors(old_a, old_b, a, b, scale)
        energy += factor_inner(terms, terms)
    if not math.isfinite(energy):
        raise FloatingPointError("Nonfinite effective weight update")
    return math.sqrt(max(0.0, energy))


def cap_multiplier(before, after, budget):
    """Choose a conservative common parameter-step multiplier in [0, 1].

    Interpolating both factors gives delta_W(t) = t * L + t**2 * Q.
    The triangle bound t * norm(L) + t**2 * norm(Q) <= budget avoids
    assuming that weight-update norm scales linearly with the parameter step.
    Norms combine every adapted layer; there is one shared multiplier.
    """
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Update budget must be finite and positive")
    proposed = update_norm(before, after)
    if proposed <= budget:
        return 1.0, {"proposed_l2": proposed, "budget_l2": budget}
    linear_energy = quadratic_energy = 0.0
    for name, (a, b, scale) in after.items():
        old_a, old_b, _ = before[name]
        da, db = a - old_a, b - old_b
        linear = [(old_a, db * scale), (da, old_b * scale)]
        quadratic = [(da, db * scale)]
        linear_energy += factor_inner(linear, linear)
        quadratic_energy += factor_inner(quadratic, quadratic)
    linear = math.sqrt(max(0.0, linear_energy))
    quadratic = math.sqrt(max(0.0, quadratic_energy))
    # Stable form of the positive quadratic root, including Q == 0.
    denominator = linear + math.sqrt(linear * linear + 4 * quadratic * budget)
    multiplier = min(1.0, 2 * budget / denominator)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise FloatingPointError("Invalid update-cap multiplier")
    return multiplier, {
        "proposed_l2": proposed,
        "budget_l2": budget,
        "linear_l2": linear,
        "quadratic_l2": quadratic,
        "conservative_bound_l2": multiplier * linear + multiplier**2 * quadratic,
    }


def interpolate_factors(before, after, multiplier):
    return {
        name: (
            old_a + multiplier * (after[name][0] - old_a),
            old_b + multiplier * (after[name][1] - old_b),
            scale,
        )
        for name, (old_a, old_b, scale) in before.items()
    }
