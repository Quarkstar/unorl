"""Audit a completed prefix of prepared-direction RL without loading weights."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def audit(run_id, step):
    if Path(run_id).name != run_id or step < 1:
        raise ValueError("Expected one run directory and a positive completed update")
    run = ROOT / "runs" / run_id
    hashes = {}

    def read(relative):
        path = run / relative
        raw = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return raw

    def rows(relative):
        # A writer may have an unfinished final record. Complete records are
        # immutable; auditing a requested prefix does not require that tail.
        raw = read(relative)
        return [json.loads(line) for line in raw.split(b"\n")[:-1] if line.strip()]

    cfg = json.loads(read("config.json"))
    if not cfg["trainer.relora_enable_merge"] or cfg["trainer.policy.model.lora.rank"] != 1:
        raise ValueError("Expected the enabled rank-one prepared recipe")
    if not 0 < step <= cfg["trainer.max_training_steps"]:
        raise ValueError("Requested update exceeds the configured experiment")
    adapter = json.loads(read("initial-adapter-audit.json"))
    assert adapter["optimizer"] == "AdamW" and adapter["rank"] == 1
    assert adapter["betas"] == [0.9, 0.999] and adapter["epsilon"] == 1e-8
    assert adapter["restart_warmup_updates"] == 0
    first = cfg["trainer.relora_first_merge_step"]
    interval = cfg["trainer.relora_merge_interval"]
    window = cfg["trainer.prepared_window_updates"]
    rho = cfg["trainer.prepared_min_total_fraction"]
    scheduled = list(range(first, step + 1, interval))
    selection = (
        [r for r in rows("prepared-selection-audit.jsonl") if r["step"] <= step]
        if scheduled
        else []
    )
    assert [r["step"] for r in selection] == scheduled
    selected_counts = {}
    boundaries = []
    for boundary in selection:
        layers = boundary["layers"]
        assert len(layers) == 252 and len({r["layer"] for r in layers}) == 252
        assert all(r["history_observations"] == window for r in layers)
        assert all(math.isfinite(r["total_descent_retained_fraction"]) for r in layers)
        assert all(r["total_descent_retained_fraction"] >= rho - 1e-6 for r in layers)
        assert all(not r["transferred"] or r["has_positive_normal_descent"] for r in layers)
        selected_counts[boundary["step"]] = sum(r["transferred"] for r in layers)
        boundaries.append(
            {
                "step": boundary["step"],
                "selected_layers": selected_counts[boundary["step"]],
                "minimum_retained_descent": min(
                    r["total_descent_retained_fraction"] for r in layers
                ),
                "minimum_reference_a_cosine": min(
                    r["reference_a_abs_cosine_at_switch"] for r in layers
                ),
                "minimum_reference_b_cosine": min(
                    r["reference_b_abs_cosine_at_switch"] for r in layers
                ),
            }
        )
    changed = [s for s in scheduled if selected_counts[s] > 0]
    memory = []
    for rank in range(8):
        rank_rows = [r for r in rows(f"training-memory-rank{rank}.jsonl") if r["step"] <= step]
        assert [r["step"] for r in rank_rows] == list(range(1, step + 1))
        assert all(
            r["rank"] == rank and 0 <= r["peak_allocated_bytes"] <= r["peak_reserved_bytes"]
            for r in rank_rows
        )
        assert [r["step"] for r in rank_rows if r["merged"]] == changed
        memory.extend(rank_rows)
    sync = [r for r in rows("backbone-sync-audit.jsonl") if r["step"] <= step]
    assert [r["step"] for r in sync] == [0] + changed
    assert all(r["base_then_adapter"] for r in sync)
    if scheduled:
        merges = [r for r in rows("merge-audit.jsonl") if r["step"] <= step]
        assert [r["step"] for r in merges] == scheduled
        assert all(r["relora/merged_layers"] == selected_counts[r["step"]] for r in merges)
    evaluations = {r["step"]: r["metrics"] for r in rows("evaluation.jsonl")}
    counts = []
    question_set = None
    for update in range(0, step + 1, 20):
        groups = defaultdict(list)
        path = f"exports/aime25/dumped_evals/global_step_{update}_evals/aime25.jsonl"
        for row in rows(path):
            assert math.isfinite(row["score"])
            groups[row["input_prompt"]].append(row["score"] > 0)
        assert len(groups) == 30 and {len(v) for v in groups.values()} == {8}
        if question_set is None:
            question_set = set(groups)
        assert set(groups) == question_set
        correct = sum(sum(v) for v in groups.values())
        solved = sum(any(v) for v in groups.values())
        assert math.isclose(correct / 240, evaluations[update]["eval/aime25/avg@8"], abs_tol=1e-12)
        assert math.isclose(solved / 30, evaluations[update]["eval/aime25/pass@8"], abs_tol=1e-12)
        counts.append({"step": update, "correct_of_240": correct, "solved_of_30": solved})
    return {
        "run_id": run_id,
        "through_step": step,
        "scheduled_switches": scheduled,
        "boundaries": boundaries,
        "training_memory_records": len(memory),
        "peak_allocated_gib": max(r["peak_allocated_bytes"] for r in memory) / 1024**3,
        "peak_reserved_gib": max(r["peak_reserved_bytes"] for r in memory) / 1024**3,
        "raw_evaluation_counts": counts,
        "source_sha256": hashes,
        "limitations": (
            "Read-only recorded-prefix audit, not a completed-run, optimizer checkpoint, "
            "live receiver tensor-equality, performance-parity or useful-rank certificate. "
            "Sources may contain later records; incomplete trailing JSONL writes are excluded. "
            "Memory is the policy allocator window, not total-device memory."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    report = audit(args.run_id, args.step)
    output = ROOT / f"research/data/{args.run_id}-recorded-prefix-step{args.step}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
