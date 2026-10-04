"""Measure hypothetical refresh loss on a real saved optimizer weight update."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

from scripts.research.snapshot import parse_jsonl
from unorl.refresh_geometry import update_space_residual
from unorl.refresh_transition import rotate_in_refresh_plane
from unorl.relora_refresh import rotated_row

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument(
        "--saved-gradual-plane",
        action="store_true",
        help="Use the active, checkpointed gradual-refresh plane instead of a hypothetical random direction",
    )
    args = parser.parse_args()
    run = ROOT / "runs" / args.run_id
    config = json.loads((run / "config.json").read_text())
    source = run / f"checkpoints/global_step_{args.step}/policy/extra_state_world_size_8_rank_0.pt"
    raw = source.read_bytes()
    state = torch.load(source, map_location="cpu", weights_only=False)
    if state["lr_scheduler"]["last_epoch"] != args.step:
        raise ValueError("Checkpoint scheduler does not match the requested update")
    factors = state["client_state"]["relora_previous_delta"]
    if not factors:
        raise ValueError("Checkpoint has no saved actual-update factors")
    scale = config["trainer.policy.model.lora.alpha"] / config["trainer.policy.model.lora.rank"]
    planes = None
    if args.saved_gradual_plane:
        transition = state["client_state"].get("gradual_refresh_transition")
        if not transition or transition.get("version") != 1 or not transition.get("count"):
            raise ValueError("Checkpoint has no active saved gradual-refresh plane")
        planes = transition["planes"]
        if planes.keys() != factors.keys():
            raise ValueError("Saved plane names differ from actual-update factors")
    angles = {}
    for angle in (0, 2, 20, 45, 60):
        results = {}
        for index, (name, terms) in enumerate(factors.items(), 1):
            if len(terms) != 2:
                raise ValueError("Expected actual update_factors representation")
            # update_factors stores (old_A, scale*dB), (dA, scale*new_B).
            a = terms[0][0] + terms[1][0]
            b = terms[1][1] / scale
            generator = torch.Generator().manual_seed(
                config["trainer.seed"] + args.step * 10000 + index
            )
            new_a = (
                rotate_in_refresh_plane(a.float(), planes[name], angle)
                if planes is not None
                else rotated_row(a.float(), angle, generator)
            ).double()
            results[name] = update_space_residual(new_a, b, terms)
        total = sum(row["target_l2"] ** 2 for row in results.values())
        unavailable = sum(row["unavailable_l2"] ** 2 for row in results.values())
        angles[str(angle)] = {
            "target_l2": math.sqrt(total),
            "unavailable_l2": math.sqrt(unavailable),
            "global_relative_residual": math.sqrt(unavailable / total),
            "mean_layer_relative_residual": statistics.mean(
                row["relative_residual"] for row in results.values()
            ),
            "layers": results,
        }
    metrics = next(
        row["metrics"]
        for row in parse_jsonl((run / "metrics.jsonl").read_bytes())
        if row["step"] == args.step
    )
    actual_norm = metrics["updates/effective_delta_l2"]
    if not math.isclose(angles["0"]["target_l2"], actual_norm, rel_tol=1e-10):
        raise ValueError("Checkpoint update factors do not match the recorded optimizer update")
    report = {
        "run_id": args.run_id,
        "step": args.step,
        "checkpoint_extra_state": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "layers": len(factors),
        "verified_actual_update_l2": actual_norm,
        "angles_degrees": angles,
        "direction_source": "active checkpointed gradual plane"
        if planes is not None
        else "seeded hypothetical random direction",
        "limitations": (
            "Hypothetical rotations of the saved post-update adapter, measured against the last actual "
            "finite optimizer weight update. No model is trained or evaluated. Zero-angle residual may "
            "be nonzero because a finite update need not lie exactly in the post-update tangent space. "
            "This measures neither the next gradient nor Adam-state transport nor a causal performance effect."
        ),
    }
    output = ROOT / f"research/data/{args.run_id}-refresh-geometry-step{args.step}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "step": args.step,
                "layers": len(factors),
                "angles": {
                    angle: result["global_relative_residual"] for angle, result in angles.items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
