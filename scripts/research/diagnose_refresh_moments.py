"""CPU proof of Adam-moment transport with recorded fixed-plane gradients."""

import json
import math
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from unorl.refresh_transition import make_refresh_plane, rotate_in_refresh_plane


class NoDenseBaseGradientProduct(TorchDispatchMode):
    """Reject base-gradient-shaped matrix products in the tiny actual branch."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        products = (
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
        )
        if func in products and tuple(result.shape[-2:]) == (11, 13):
            raise AssertionError("Actual frozen-base branch formed a dense gradient-shaped product")
        return result


def components(a, gradient, plane):
    """Three output vectors whose first two sum to the rank-one B gradient."""
    first, second = plane
    c1, c2 = (a * first).sum(), (a * second).sum()
    off_plane = a - c1 * first - c2 * second
    g1, g2 = gradient @ first.T, gradient @ second.T
    return torch.stack((gradient @ off_plane.T, c1 * g1 + c2 * g2, c1 * g2 - c2 * g1)).squeeze(-1)


def transport(first_moment, cross_moment, angle_degrees):
    """Rotate the plane-gradient components and their raw cross moments."""
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    transform = first_moment.new_tensor([[1, 0, 0], [0, cosine, sine], [0, -sine, cosine]])
    return transform @ first_moment, torch.einsum(
        "ij,jko,lk->ilo", transform, cross_moment, transform
    )


def b_moments(first_moment, cross_moment):
    return first_moment[0] + first_moment[1], cross_moment[0, 0] + cross_moment[
        1, 1
    ] + 2 * cross_moment[0, 1]


def direct_history_moments(history, beta1, beta2):
    first = torch.zeros_like(history[0][1][:, 0])
    second = torch.zeros_like(first)
    for a, gradient in history:
        value = (gradient @ a.T).squeeze(-1)
        first = beta1 * first + (1 - beta1) * value
        second = beta2 * second + (1 - beta2) * value.square()
    return first, second


def verify_projected_gradient_collection():
    """Toy output hook records two projections while the base weight is frozen."""
    generator = torch.Generator().manual_seed(123)
    dtype = torch.float64
    try:
        with NoDenseBaseGradientProduct():
            torch.zeros(11, 2, dtype=dtype) @ torch.zeros(2, 13, dtype=dtype)
    except AssertionError:
        pass
    else:
        raise AssertionError("The dense-product guard failed its negative control")
    base = torch.nn.Parameter(
        torch.randn(11, 13, generator=generator, dtype=dtype), requires_grad=False
    )
    a = torch.nn.Parameter(0.2 * torch.randn(1, 13, generator=generator, dtype=dtype))
    b = torch.nn.Parameter(0.02 * torch.randn(11, 1, generator=generator, dtype=dtype))
    plane = make_refresh_plane(a, seed=17)
    scale = 32
    collected = torch.zeros(11, 2, dtype=dtype)
    oracle_gradient = torch.zeros_like(base)
    oracle_a, oracle_b = torch.zeros_like(a), torch.zeros_like(b)
    tokens = (6, 10, 4, 7)
    input_gradient_error = 0.0
    for index, count in enumerate(tokens):
        shape = (2, count // 2, 13) if count % 2 == 0 else (count, 13)
        x = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=index % 2 == 1)
        target = torch.randn(*shape[:-1], 11, generator=generator, dtype=dtype)
        # The additional hook retains only two projected values per input token.
        projected = x.detach().reshape(-1, 13) @ torch.cat(plane).T

        @torch.no_grad()
        def capture(gradient, projected=projected):
            collected.add_(scale * gradient.reshape(-1, 11).T @ projected)

        with NoDenseBaseGradientProduct():
            output = x @ base.T + scale * (x @ a.T) @ b.T
            output.register_hook(capture)
            ((output - target).square().mean() / len(tokens)).backward()
        assert base.grad is None
        # Allocate a dense reference gradient only in this separate tiny oracle.
        reference_weight = base.detach().clone().requires_grad_()
        reference_a = a.detach().clone().requires_grad_()
        reference_b = b.detach().clone().requires_grad_()
        reference_x = x.detach().clone().requires_grad_(x.requires_grad)
        reference_output = reference_x @ (reference_weight + scale * reference_b @ reference_a).T
        reference_loss = (reference_output - target).square().mean() / len(tokens)
        parameters = (reference_weight, reference_a, reference_b)
        if x.requires_grad:
            parameters += (reference_x,)
        gradients = torch.autograd.grad(reference_loss, parameters)
        g, ga, gb = gradients[:3]
        if x.requires_grad:
            torch.testing.assert_close(x.grad, gradients[3], atol=1e-12, rtol=1e-12)
            input_gradient_error = max(
                input_gradient_error, (x.grad - gradients[3]).abs().max().item()
            )
        oracle_gradient += g
        oracle_a += ga
        oracle_b += gb
    expected = scale * oracle_gradient @ torch.cat(plane).T
    torch.testing.assert_close(collected, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(a.grad, oracle_a, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(b.grad, oracle_b, atol=1e-12, rtol=1e-12)
    return {
        "scope": "Float64 CPU frozen 11x13 linear layer, rank-one trainable factors, four accumulated microbatches; not PEFT/FSDP or checkpoint-recomputed execution.",
        "tokens": list(tokens),
        "trainable_parameters": a.numel() + b.numel(),
        "frozen_base_gradient_allocated": base.grad is not None,
        "dense_base_gradient_product_guard_passed": True,
        "projected_gradient_max_abs_error": (collected - expected).abs().max().item(),
        "native_a_gradient_max_abs_error": (a.grad - oracle_a).abs().max().item(),
        "native_b_gradient_max_abs_error": (b.grad - oracle_b).abs().max().item(),
        "native_input_gradient_max_abs_error": input_gradient_error,
        "input_gradient_cases": [False, True],
        "variable_two_and_three_dimensional_inputs": True,
        "projected_values_per_token": 2,
        "largest_projection_tensor_values": 2 * max(tokens),
        "full_dense_gradient_used_only_in_reference": True,
        "memory_scope": "Additional hook payload contains the projected input tensor; native LoRA activation storage and end-to-end peak memory are not measured.",
    }


def main():
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(1, 13, generator=generator, dtype=torch.float64)
    plane = make_refresh_plane(a, seed=17)
    first = torch.zeros(3, 11, dtype=a.dtype)
    cross = torch.zeros(3, 3, 11, dtype=a.dtype)
    beta1, beta2 = 0.9, 0.999
    history, boundaries = [], []
    worst_first_error = worst_second_error = 0.0
    for step in range(1, 41):
        # Toy dense gradients are used only as an oracle, never as a proposed
        # production storage scheme. A varies through ordinary learning.
        a += 0.03 * torch.randn(a.shape, generator=generator, dtype=a.dtype)
        gradient = 32 * torch.randn(11, 13, generator=generator, dtype=a.dtype)
        values = components(a, gradient, plane)
        torch.testing.assert_close(
            values[0] + values[1], (gradient @ a.T).squeeze(-1), atol=1e-12, rtol=1e-12
        )
        first = beta1 * first + (1 - beta1) * values
        cross = beta2 * cross + (1 - beta2) * torch.einsum("io,jo->ijo", values, values)
        history.append((a.clone(), gradient))
        if step in (20, 40):
            old_first, old_second = b_moments(first, cross)
            new_a = rotate_in_refresh_plane(a, plane, 20)
            actual_cosine = (a * new_a).sum() / (a.norm() * new_a.norm())
            first, cross = transport(first, cross, 20)
            history = [(rotate_in_refresh_plane(row, plane, 20), g) for row, g in history]
            a = new_a
            target_first, target_second = direct_history_moments(history, beta1, beta2)
            exact_first, exact_second = b_moments(first, cross)
            boundaries.append(
                {
                    "step": step,
                    "plane_angle_degrees": 20,
                    "actual_row_cosine": actual_cosine.item(),
                    "exact_first_max_abs_error": (exact_first - target_first).abs().max().item(),
                    "exact_second_max_abs_error": (exact_second - target_second).abs().max().item(),
                    "cosine_only_first_relative_error": (
                        (actual_cosine * old_first - target_first).norm() / target_first.norm()
                    ).item(),
                    "retained_variance_relative_error": (
                        (old_second - target_second).norm() / target_second.norm()
                    ).item(),
                    "gradient_history_entries": len(history),
                }
            )
        target_first, target_second = direct_history_moments(history, beta1, beta2)
        projected_first, projected_second = b_moments(first, cross)
        torch.testing.assert_close(projected_first, target_first, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(projected_second, target_second, atol=1e-12, rtol=1e-12)
        worst_first_error = max(
            worst_first_error, (projected_first - target_first).abs().max().item()
        )
        worst_second_error = max(
            worst_second_error, (projected_second - target_second).abs().max().item()
        )
    report = {
        "scope": "Synthetic float64 CPU proof with varying A and fixed historical dense gradients; not an RL experiment or production gradient collector.",
        "seed": 42,
        "betas": [beta1, beta2],
        "updates": 40,
        "boundaries": boundaries,
        "worst_first_max_abs_error": worst_first_error,
        "worst_second_max_abs_error": worst_second_error,
        "projected_gradient_collection": verify_projected_gradient_collection(),
        "stored_output_vectors": {
            "first_moments": 3,
            "full_cross_moments": 9,
            "symmetric_cross_moments_possible": 6,
        },
        "requirements": [
            "Select a fixed plane before collecting the optimizer history, then retain it across rotations.",
            "Record two extra projected dense gradients per adapter layer, without forming a dense weight gradient in production.",
            "Apply the same loss scaling, accumulation, distributed reduction, and gradient-clipping factor to all components.",
            "Preserve Adam counters; transport does not add an optimizer update.",
        ],
        "limitations": "Exactness concerns counterfactual projected-gradient EMAs with historical dense gradients and their scaling held fixed. It does not recreate alternative rollouts, dense gradients, or the global clipping factor of a counterfactual training run. A newly chosen plane cannot recover unrecorded earlier directions. A toy CPU output hook validates gradient collection; PEFT/FSDP integration, checkpoint recomputation, mixed precision, activation costs, peak memory and performance remain untested. The prepared ten-increment trial is unchanged.",
    }
    output = (
        Path(__file__).resolve().parents[2] / "research/data/refresh-moment-transport-analysis.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("requirements", "limitations")
            }
        )
    )


if __name__ == "__main__":
    main()
