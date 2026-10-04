"""Read-only partial checkpoint audit for scheduled gradual-refresh milestones."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from scripts.research.analyze_refresh_factor_span import factor_geometry
from scripts.research.snapshot import parse_jsonl
from unorl.refresh_transition import rotate_in_refresh_plane

ROOT = Path(__file__).resolve().parents[2]


def audit(run_id, step):
    if Path(run_id).name != run_id or step <= 0:
        raise ValueError("Expected one run directory and a positive checkpoint step")
    run = ROOT / "runs" / run_id
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return raw

    cfg = json.loads(read(run / "config.json"))
    first = cfg["trainer.relora_first_merge_step"]
    interval = cfg["trainer.relora_merge_interval"]
    increments = cfg["trainer.relora_refresh_updates"]
    angle = cfg["trainer.relora_refresh_angle_degrees"] / increments
    if not cfg["trainer.relora_enable_merge"] or cfg["trainer.policy.model.lora.rank"] != 1:
        raise ValueError("Expected the enabled rank-one gradual-refresh recipe")
    scheduled = [
        s for s in range(1, step + 1) if s >= first and (s - first) % interval < increments
    ]
    offset = (step - first) % interval if step >= first else None
    active = offset is not None and offset < increments - 1
    expected_count = offset + 1 if active else 0
    expected_start = step - offset if active else None
    checkpoint = run / f"checkpoints/global_step_{step}/policy"
    rank_checks = []
    reference_planes = None
    rank0_client = None
    for rank in range(8):
        extra_path = checkpoint / f"extra_state_world_size_8_rank_{rank}.pt"
        extra = torch.load(extra_path, map_location="cpu", weights_only=False)
        read(extra_path)
        assert extra["lr_scheduler"]["last_epoch"] == step
        client = extra["client_state"]
        transition = client["gradual_refresh_transition"]
        assert transition["version"] == 1
        assert transition["schedule"] == {
            "first": first,
            "interval": interval,
            "updates": increments,
            "total_angle": cfg["trainer.relora_refresh_angle_degrees"],
            "enabled": True,
        }
        assert transition["count"] == expected_count and transition["start"] == expected_start
        planes = transition["planes"]
        if active:
            assert len(planes) == 252
            for plane in planes.values():
                rotate_in_refresh_plane(plane[0], plane, 0)
            if reference_planes is None:
                reference_planes = planes
            else:
                assert planes.keys() == reference_planes.keys()
                assert all(
                    torch.equal(a, b)
                    for name in planes
                    for a, b in zip(planes[name], reference_planes[name])
                )
            expected_names = set(planes) if rank == 0 else set()
            assert set(transition["columns"]) == expected_names
            assert set(transition["history_positions"]) == expected_names
        else:
            assert not any(transition[key] for key in ("planes", "columns", "history_positions"))
        if rank == 0:
            rank0_client = client
        optimizer_path = checkpoint / f"optim_world_size_8_rank_{rank}.pt"
        optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        read(optimizer_path)
        for group in optimizer["param_groups"]:
            assert math.isclose(
                group["lr"], cfg["trainer.policy.optimizer_config.lr"], rel_tol=1e-12
            )
            assert group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8
            assert group["weight_decay"] == 0
        counters = [state["step"].item() for state in optimizer["state"].values() if state]
        assert len(counters) == 504 and set(counters) == {float(step)}
        rank_checks.append(
            {
                "rank": rank,
                "adam_active_states": len(counters),
                "adam_counter": step,
                "transition_count": expected_count,
            }
        )
    history = rank0_client["relora_factor_history"]
    deltas = rank0_client["relora_previous_delta"]
    assert len(history) == 252 and history.keys() == deltas.keys()
    started_cycles = len({s - (s - first) % interval for s in scheduled})
    layers = {}
    for name, terms in deltas.items():
        assert len(terms) == 2 and len(history[name]) == 2 * started_cycles
        a = terms[0][0] + terms[1][0]
        if active:
            transition = rank0_client["gradual_refresh_transition"]
            position = transition["history_positions"][name]
            assert position + 2 == len(history[name])
            for pair, row, column in zip(
                history[name][position:], transition["planes"][name], transition["columns"][name]
            ):
                assert torch.equal(pair[0], row) and torch.equal(pair[1], column)
            a = rotate_in_refresh_plane(a.float(), transition["planes"][name], angle)
        layers[name] = factor_geometry(history[name] + [(a, terms[1][1])])
    memory = []
    for rank in range(8):
        rows = [
            row
            for row in parse_jsonl(read(run / f"training-memory-rank{rank}.jsonl"))
            if row["step"] <= step
        ]
        assert len(rows) == step and {r["step"] for r in rows} == set(range(1, step + 1))
        assert all(
            r["rank"] == rank and 0 <= r["peak_allocated_bytes"] <= r["peak_reserved_bytes"]
            for r in rows
        )
        assert sorted(r["step"] for r in rows if r["merged"]) == scheduled
        memory.extend(rows)
    transfers = [
        row for row in parse_jsonl(read(run / "backbone-sync-audit.jsonl")) if row["step"] <= step
    ]
    assert [r["step"] for r in transfers] == [0] + scheduled
    assert all(r["base_then_adapter"] for r in transfers)
    merges = [r for r in parse_jsonl(read(run / "merge-audit.jsonl")) if r["step"] <= step]
    assert [r["step"] for r in merges] == scheduled
    evaluations = {r["step"]: r["metrics"] for r in parse_jsonl(read(run / "evaluation.jsonl"))}
    counts = []
    question_set = None
    for s in range(0, step + 1, 20):
        groups = defaultdict(list)
        path = run / f"exports/aime25/dumped_evals/global_step_{s}_evals/aime25.jsonl"
        for row in parse_jsonl(read(path)):
            assert math.isfinite(row["score"])
            groups[row["input_prompt"]].append(row["score"] > 0)
        assert len(groups) == 30 and {len(values) for values in groups.values()} == {8}
        if question_set is None:
            question_set = set(groups)
        assert set(groups) == question_set
        correct = sum(sum(values) for values in groups.values())
        solved = sum(any(values) for values in groups.values())
        assert math.isclose(correct / 240, evaluations[s]["eval/aime25/avg@8"], abs_tol=1e-12)
        assert math.isclose(solved / 30, evaluations[s]["eval/aime25/pass@8"], abs_tol=1e-12)
        counts.append(
            {"step": s, "correct": correct, "responses": 240, "solved": solved, "questions": 30}
        )
    allocated = max(memory, key=lambda r: r["peak_allocated_bytes"])
    reserved = max(memory, key=lambda r: r["peak_reserved_bytes"])
    return {
        "run_id": run_id,
        "through_step": step,
        "completed_run": False,
        "checkpoint_rank_checks": rank_checks,
        "scheduled_refresh_steps": scheduled,
        "active_transition": active,
        "mean_factor_geometry": {
            key: statistics.mean(r[key] for r in layers.values())
            for key in next(iter(layers.values()))
        },
        "layers": layers,
        "training_memory_records": len(memory),
        "max_allocated_gib": allocated["peak_allocated_bytes"] / 1024**3,
        "max_reserved_gib": reserved["peak_reserved_bytes"] / 1024**3,
        "max_allocated_at": {"step": allocated["step"], "rank": allocated["rank"]},
        "evaluation_counts": counts,
        "source_sha256": hashes,
        "limitations": "Read-only partial checkpoint audit, not a completed-run audit or full-model resume. Spectrum excludes base-rounding residuals. Source logs can contain later records; checks cover only the requested prefix. Transfer completion does not independently certify server tensor equality. Memory is training allocator only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = audit(args.run_id, args.step)
    output = ROOT / f"research/data/{args.run_id}-checkpoint-step{args.step}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"layers", "source_sha256", "checkpoint_rank_checks"}
            }
        )
    )


if __name__ == "__main__":
    main()
