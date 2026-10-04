"""Read-only native checkpoint audit for prepared-direction milestones."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

from scripts.research.audit_prepared_run import audit as audit_prefix
from unorl.prepared_moments import PreparedBasisMoments

ROOT = Path(__file__).resolve().parents[2]


def audit(run_id, step):
    prefix = audit_prefix(run_id, step)
    run = ROOT / "runs" / run_id
    cfg = json.loads((run / "config.json").read_bytes())
    first = cfg["trainer.relora_first_merge_step"]
    interval = cfg["trainer.relora_merge_interval"]
    window = cfg["trainer.prepared_window_updates"]
    protocol = {
        "version": 1,
        "seed": cfg["trainer.seed"],
        "first_merge": first,
        "merge_interval": interval,
        "window_updates": window,
        "min_total_fraction": cfg["trainer.prepared_min_total_fraction"],
        "angles": cfg["trainer.prepared_angles"],
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "constant_lr": cfg["trainer.policy.optimizer_config.lr"],
    }
    future = range(first, cfg["trainer.max_training_steps"] + 1, interval)
    active_target = next((s for s in future if s - window < step < s), None)
    selections = []
    selection_path = run / "prepared-selection-audit.jsonl"
    if selection_path.exists():
        selections = [json.loads(r) for r in selection_path.read_bytes().split(b"\n")[:-1] if r]
        selections = [r for r in selections if r["step"] <= step]
    selected_at = {}
    selection_counts = Counter()
    for boundary in selections:
        for row in boundary["layers"]:
            if row["transferred"]:
                selected_at[row["layer"]] = boundary["step"]
                selection_counts[row["layer"]] += 1
    checkpoint = run / f"checkpoints/global_step_{step}/policy"
    rank_checks = []
    for rank in range(8):

        def load(filename):
            path = checkpoint / filename
            raw = path.read_bytes()
            prefix["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
            return torch.load(path, map_location="cpu", weights_only=False)

        extra = load(f"extra_state_world_size_8_rank_{rank}.pt")
        assert extra["rank"] == rank and extra["world_size"] == 8
        assert extra["lr_scheduler"]["last_epoch"] == step
        client = extra["client_state"]
        assert client["prepared_protocol"] == protocol
        counters = client["native_adam_steps"]
        assert len(counters) == 504
        layers = set()
        for name, counter in counters.items():
            matched = [
                suffix
                for suffix in (".lora_A.default.weight", ".lora_B.default.weight")
                if name.endswith(suffix)
            ]
            assert len(matched) == 1
            layer = name.removesuffix(matched[0])
            layers.add(layer)
            last = selected_at.get(layer)
            expected = step if last is None else window + step - last
            assert counter == expected, (rank, name, counter, expected)
        assert len(layers) == 252
        prepared = client["prepared"]
        if active_target is None:
            assert not prepared and client["prepared_start"] is None
            assert client["prepared_target"] is None
        else:
            assert client["prepared_target"] == active_target
            assert client["prepared_start"] == active_target - window
            assert set(prepared) == layers
            for saved in prepared.values():
                observer = PreparedBasisMoments.from_state_dict(saved, device="cpu")
                assert observer.count == step - (active_target - window)
                assert observer.pending_microbatches == 0
                assert observer.beta1 == 0.9 and observer.beta2 == 0.999
                assert not observer.ga.any() and not observer.gb.any()
        history = client["relora_factor_history"]
        if rank == 0:
            assert set(history) <= layers
            assert all(len(history.get(name, [])) == 2 * selection_counts[name] for name in layers)
            assert set(client["relora_previous_delta"]) == layers
        else:
            assert not history
            assert client["relora_previous_delta"] is None
        optimizer = load(f"optim_world_size_8_rank_{rank}.pt")
        assert len(optimizer["param_groups"]) == 1
        group = optimizer["param_groups"][0]
        assert math.isclose(group["lr"], protocol["constant_lr"], rel_tol=1e-12)
        assert list(group["betas"]) == protocol["betas"]
        assert group["eps"] == protocol["epsilon"] and group["weight_decay"] == 0
        states = [state for state in optimizer["state"].values() if state]
        assert len(states) == 504
        actual = [float(state["step"]) for state in states]
        assert Counter(actual) == Counter(counters.values())
        assert all(set(state) == {"step", "exp_avg", "exp_avg_sq"} for state in states)
        for state in states:
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = state[key]
                local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
                assert torch.isfinite(local).all(), (rank, key, "nonfinite moment")
                if key == "exp_avg_sq":
                    assert (local >= 0).all(), (rank, key, "negative second moment")
        rank_checks.append(
            {
                "rank": rank,
                "active_optimizer_states": len(states),
                "finite_moment_states": len(states),
                "counter_histogram": dict(Counter(actual)),
                "active_preparation_target": active_target,
                "preparation_count": step - (active_target - window) if active_target else 0,
            }
        )
    prefix["checkpoint_rank_checks"] = rank_checks
    prefix["checkpoint_protocol"] = protocol
    prefix["limitations"] += (
        " Native checkpoint checks cover scheduler/protocol, per-name saved counters, "
        "optimizer counter multisets, finite local moments, preparation buffers and correction-history lengths. "
        "They do not associate optimizer integer IDs with parameter names, reconstruct "
        "full model weights, test full-model resume or prove useful learned rank."
    )
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = audit(args.run_id, args.step)
    output = ROOT / f"research/data/{args.run_id}-checkpoint-step{args.step}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
