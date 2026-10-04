"""Separate input/output alignment in a saved accumulated low-rank update."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from safetensors.torch import load

ROOT = Path(__file__).resolve().parents[2]


def factor_geometry(pairs):
    """Measure rank and residual energy without constructing the dense update.

    The final pair is the active adapter. Earlier pairs are frozen correction
    history. Normalize directions only for span diagnostics; retain all scales
    when measuring the actual accumulated update and its residual energies.
    """
    a = torch.cat([row.double() for row, _ in pairs], dim=0)
    b = torch.cat([column.double() for _, column in pairs], dim=1)
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    compact = rb @ ra.T
    energy = compact.square().sum()
    if energy <= 0:
        raise ValueError("An accumulated nonzero update is required")
    active_a, active_b = (value.double() for value in pairs[-1])
    unit_b = active_b / active_b.norm()
    unit_a = active_a.T / active_a.norm()
    output_projection = unit_b.T @ qb
    input_projection = unit_a.T @ qa
    output_residual = compact - output_projection.T @ (output_projection @ compact)
    input_residual = compact - (compact @ input_projection.T) @ input_projection
    singular = torch.linalg.svdvals(compact)

    def direction_rank(matrix):
        normalized = matrix / matrix.norm(dim=0, keepdim=True).clamp_min(1e-30)
        values = torch.linalg.svdvals(normalized)
        return (values.square().sum() / values[0].square()).item()

    return {
        "accumulated_stable_rank": (energy / singular[0].square()).item(),
        "energy_outside_active_output_direction": (output_residual.square().sum() / energy).item(),
        "energy_outside_active_input_direction": (input_residual.square().sum() / energy).item(),
        "normalized_output_direction_stable_rank": direction_rank(b),
        "normalized_input_direction_stable_rank": direction_rank(a.T),
        "sum_component_norms_over_accumulated_norm": (
            sum(row.double().norm() * col.double().norm() for row, col in pairs) / energy.sqrt()
        ).item(),
        "accumulated_update_l2": energy.sqrt().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = ROOT / "runs" / args.run_id
    config = json.loads((run / "config.json").read_text())
    source = run / f"checkpoints/global_step_{args.step}/policy/extra_state_world_size_8_rank_0.pt"
    raw = source.read_bytes()
    state = torch.load(source, map_location="cpu", weights_only=False)
    if state["lr_scheduler"]["last_epoch"] != args.step:
        raise ValueError("Requested step differs from checkpoint scheduler")
    client = state["client_state"]
    if "relora_factor_history" not in client and config.get("trainer.relora_enable_merge"):
        raise ValueError("Merged checkpoints require their saved correction history")
    history = client.get("relora_factor_history", {})
    if config["trainer.policy.model.lora.rank"] != 1:
        raise ValueError("Active-direction diagnostics require rank-one adapters")
    scale = config["trainer.policy.model.lora.alpha"] / config["trainer.policy.model.lora.rank"]
    if scale <= 0:
        raise ValueError("Adapter scale must be positive")
    adapter_source = source.parent / "lora_adapter/adapter_model.safetensors"
    adapter_raw = adapter_source.read_bytes()
    adapter = load(adapter_raw)
    suffixes = (".lora_A.weight", ".lora_B.weight")
    if any(not name.endswith(suffixes) for name in adapter):
        raise ValueError("Expected only A/B tensors in the checkpoint adapter export")
    names = {name.removesuffix(suffixes[0]) for name in adapter if name.endswith(suffixes[0])}
    if not names or len(adapter) != 2 * len(names) or not history.keys() <= names:
        raise ValueError("Correction history and saved adapter names differ")
    layers = {}
    for name in sorted(names):
        active_a = adapter[name + suffixes[0]]
        active_b = adapter[name + suffixes[1]]
        if (
            active_a.ndim != 2
            or active_a.shape[0] != 1
            or active_b.ndim != 2
            or active_b.shape[1] != 1
        ):
            raise ValueError("Expected saved rank-one factors")
        # The latest optimizer delta precedes a scheduled reparameterization.
        # Read the actual saved adapter, which is valid at the boundary too.
        layers[name] = factor_geometry(history.get(name, []) + [(active_a, active_b * scale)])
    means = {
        key: statistics.mean(row[key] for row in layers.values())
        for key in next(iter(layers.values()))
    }
    report = {
        "run_id": args.run_id,
        "step": args.step,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source": str(source.relative_to(ROOT)),
        "active_factor_source": str(adapter_source.relative_to(ROOT)),
        "active_factor_source_sha256": hashlib.sha256(adapter_raw).hexdigest(),
        "layers": layers,
        "mean_layer_metrics": means,
        "global_update_l2": sum(row["accumulated_update_l2"] ** 2 for row in layers.values())
        ** 0.5,
        "limitations": "Saved factor geometry only; excludes FP32 base-rounding residuals. Normalized span ranks discard component magnitudes. No future-gradient, Adam-history, causal accuracy or performance claim.",
    }
    output = ROOT / f"research/data/{args.run_id}-factor-span-step{args.step}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "step": args.step,
                "layers": len(layers),
                "mean_layer_metrics": means,
                "global_update_l2": report["global_update_l2"],
            }
        )
    )


if __name__ == "__main__":
    main()
