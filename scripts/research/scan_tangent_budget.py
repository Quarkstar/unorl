"""Scan hypothetical adapter rotations against a saved actual optimizer delta.

CPU-only geometry: candidate directions are fresh seeded random directions,
not the production selector or observed optimizer moments. The budget bounds
the best linear tangent approximation, not actual transported Adam updates.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
from safetensors.torch import load

ROOT = Path(__file__).resolve().parents[2]


def residual_energies(a, b, u0, u1, v0, v1, angles):
    """Evaluate both-sided projected norms from small Gram matrices."""
    aa, bb = a @ a.T, b.T @ b
    energy = (aa * bb).sum().item()
    ub = angles.cos()[:, None] * (u0 @ b) + angles.sin()[:, None] * (u1 @ b)
    av = angles.cos()[:, None] * (a @ v0) + angles.sin()[:, None] * (a @ v1)
    br = bb[None] - ub[:, :, None] * ub[:, None, :]
    ar = aa[None] - av[:, :, None] * av[:, None, :]
    return energy, (br * ar).sum(dim=(1, 2))


def validate_compact_scan():
    """Compare the compact formula with independent tiny dense projections."""
    angles = torch.tensor([0.0, 0.3, 1.5], dtype=torch.float64)
    errors = []
    for seed in (17, 29, 43, 71):
        generator = torch.Generator().manual_seed(seed)
        a = torch.randn(2, 11, generator=generator, dtype=torch.float64)
        b = torch.randn(7, 2, generator=generator, dtype=torch.float64)
        u, _ = torch.linalg.qr(torch.randn(7, 2, generator=generator, dtype=torch.float64))
        v, _ = torch.linalg.qr(torch.randn(11, 2, generator=generator, dtype=torch.float64))
        energy, residual = residual_energies(a, b, u[:, 0], u[:, 1], v[:, 0], v[:, 1], angles)
        dense = b @ a
        if abs(energy - dense.square().sum().item()) > 1e-10:
            raise AssertionError("Compact update norm differs from dense norm")
        for index, angle in enumerate(angles):
            un = angle.cos() * u[:, 0] + angle.sin() * u[:, 1]
            vn = angle.cos() * v[:, 0] + angle.sin() * v[:, 1]
            normal = (
                (torch.eye(7) - torch.outer(un, un)) @ dense @ (torch.eye(11) - torch.outer(vn, vn))
            )
            errors.append(abs(residual[index].item() - normal.square().sum().item()))
    if max(errors) > 1e-10:
        raise AssertionError("Compact rotation scan differs from dense projection")
    return {"cases": len(errors), "maximum_absolute_squared_norm_error": max(errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    validation = validate_compact_scan()
    root = ROOT / "runs" / args.run_id / f"checkpoints/global_step_{args.step}/policy"
    sources = (
        root / "extra_state_world_size_8_rank_0.pt",
        root / "lora_adapter/adapter_model.safetensors",
    )
    raw = [path.read_bytes() for path in sources]
    state = torch.load(sources[0], map_location="cpu", weights_only=False)
    if state["lr_scheduler"]["last_epoch"] != args.step:
        raise ValueError("Requested checkpoint differs from saved scheduler")
    delta = state["client_state"]["relora_previous_delta"]
    adapter = load(raw[1])
    angles = torch.linspace(0, math.pi / 2, 181, dtype=torch.float64)
    budgets = (0.1, 0.25, 0.5)
    layers = {}
    total_energy = 0.0
    total_residual = torch.zeros_like(angles)
    for index, (name, pairs) in enumerate(sorted(delta.items())):
        generator = torch.Generator().manual_seed(42 + index)

        def basis(factor):
            first = factor.double().reshape(-1)
            first /= first.norm()
            extra = torch.randn(first.shape, generator=generator, dtype=torch.float64)
            extra -= first * (first @ extra)
            return first, extra / extra.norm()

        v0, v1 = basis(adapter[name + ".lora_A.weight"])
        u0, u1 = basis(adapter[name + ".lora_B.weight"])
        a = torch.cat([row.double() for row, _ in pairs], dim=0)
        b = torch.cat([column.double() for _, column in pairs], dim=1)
        energy, residual = residual_energies(a, b, u0, u1, v0, v1, angles)
        if not math.isfinite(energy) or energy <= 0:
            raise ValueError("Expected a finite nonzero recorded delta")
        if (residual < -1e-12 * energy).any() or not torch.isfinite(residual).all():
            raise ValueError("Invalid projected residual energy")
        residual = residual.clamp_min(0)
        relative = (residual / energy).sqrt()

        def connected_max(values, budget):
            failures = torch.where(values > budget)[0]
            last = int(failures[0]) - 1 if len(failures) else len(angles) - 1
            return None if last < 0 else float(angles[last] * 180 / math.pi)

        layers[name] = {
            "baseline_relative_tangent_error": relative[0].item(),
            "connected_feasible_max_degrees": {
                str(budget): connected_max(relative, budget) for budget in budgets
            },
        }
        total_energy += energy
        total_residual += residual
    summary = {}
    for budget in budgets:
        values = [row["connected_feasible_max_degrees"][str(budget)] for row in layers.values()]
        if any(value is None for value in values):
            raise ValueError("At least one unrotated adapter already exceeds the budget")
        summary[str(budget)] = {
            "minimum_max_degrees": min(values),
            "median_max_degrees": statistics.median(values),
            "maximum_max_degrees": max(values),
            "global_common_angle_max_degrees": connected_max(
                (total_residual / total_energy).sqrt(), budget
            ),
        }
    report = {
        "run_id": args.run_id,
        "step": args.step,
        "seed": 42,
        "compact_formula_validation": validation,
        "candidate_scope": "Independent random orthogonal directions indexed by sorted layer names; not production window candidates",
        "grid_degrees": [float(value * 180 / math.pi) for value in angles],
        "global_relative_tangent_error": (total_residual / total_energy).sqrt().tolist(),
        "budget_summary": summary,
        "layers": layers,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(value).hexdigest()
            for path, value in zip(sources, raw)
        },
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": "Hypothetical equal-angle rotations of both factors. Maximum angles are the last feasible 0.5-degree grid point connected to zero. These are necessary geometric bounds using a recorded finite update, not actual Adam-state transport, future gradient alignment, finite-step continuity, learned rank, or RL performance. The live run is unchanged.",
    }
    output = ROOT / "research/data" / f"{args.run_id}-tangent-budget-step{args.step}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
