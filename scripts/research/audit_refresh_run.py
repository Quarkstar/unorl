"""Audit completed refresh experiments without reading or retaining weights."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.research.snapshot import parse_jsonl


def audit_run(root, run_id, questions=30):
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Run ID must name one run directory")
    run = root / "runs" / run_id
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return raw

    exit_path = root / f"runs/logs/{run_id}.exit-status"
    if not exit_path.exists() or int(read(exit_path)) != 0:
        raise ValueError("Run must have a recorded successful exit")
    cfg = json.loads(read(run / "config.json"))
    target = cfg["trainer.max_training_steps"]
    recipe = {
        "trainer.algorithm.advantage_estimator": "grpo",
        "trainer.algorithm.policy_loss_type": "regular",
        "trainer.algorithm.loss_reduction": "token_mean",
        "trainer.algorithm.use_kl_loss": False,
        "trainer.algorithm.use_kl_in_reward": False,
        "trainer.policy.model.lora.rank": 1,
        "trainer.policy.model.lora.alpha": 32,
        "trainer.policy.optimizer_config.lr": 1.5e-5,
        "trainer.policy.optimizer_config.num_warmup_steps": 0,
        "trainer.policy.optimizer_config.scheduler": "constant",
        "trainer.policy.optimizer_config.weight_decay": 0,
        "trainer.train_batch_size": 32,
        "generator.n_samples_per_prompt": 8,
        "generator.sampling_params.max_generate_length": 8192,
    }
    mismatches = [key for key, value in recipe.items() if cfg.get(key) != value]
    if mismatches:
        raise ValueError("Run differs from the matched recipe: " + ", ".join(mismatches))
    if cfg.get("trainer.resume_path"):
        raise ValueError("This audit expects a from-base run")
    if cfg["generator.eval_n_samples_per_prompt"] != 8:
        raise ValueError("This audit expects AIME25 avg@8/pass@8")
    expected = set(range(1, target + 1))
    metrics = {}
    for row in parse_jsonl(read(run / "metrics.jsonl")):
        metrics.setdefault(row["step"], {}).update(row["metrics"])
    training = {step: row for step, row in metrics.items() if "reward/mean_positive_reward" in row}
    if set(training) != expected:
        raise ValueError("Training metric steps do not cover the exact configured budget")
    for row in training.values():
        for key in (
            "reward/mean_positive_reward",
            "policy/policy_entropy",
            "policy/grad_norm",
            "generate/avg_num_tokens",
            "updates/effective_delta_l2",
        ):
            if key not in row or not math.isfinite(row[key]):
                raise ValueError(f"Missing or nonfinite training metric: {key}")
        if not 0 <= row["reward/mean_positive_reward"] <= 1:
            raise ValueError("Training correctness is outside [0, 1]")
    first = cfg.get("trainer.relora_first_merge_step")
    interval = cfg["trainer.relora_merge_interval"]
    first = interval if first is None else first
    increments = cfg.get("trainer.relora_refresh_updates", 1)
    merge_steps = sorted(
        step
        for step in expected
        if cfg["trainer.relora_enable_merge"]
        and step >= first
        and (step - first) % interval < increments
    )
    memory = []
    for rank in range(cfg["trainer.placement.policy_num_gpus_per_node"]):
        rows = parse_jsonl(read(run / f"training-memory-rank{rank}.jsonl"))
        if len(rows) != target or {row["step"] for row in rows} != expected:
            raise ValueError(f"Rank {rank} memory records do not cover every update exactly once")
        if any(row["rank"] != rank for row in rows):
            raise ValueError("Memory record rank disagrees with its filename")
        if sorted(row["step"] for row in rows if row["merged"]) != merge_steps:
            raise ValueError(f"Rank {rank} applied refresh schedule differs from configuration")
        for row in rows:
            allocated, reserved = row["peak_allocated_bytes"], row["peak_reserved_bytes"]
            if not 0 <= allocated <= reserved or not all(
                math.isfinite(value) for value in (allocated, reserved)
            ):
                raise ValueError("Invalid training allocator memory measurement")
        memory.extend(rows)
    evaluations = {}
    for row in parse_jsonl(read(run / "evaluation.jsonl")):
        evaluations.setdefault(row["step"], {}).update(row["metrics"])
    eval_steps = list(range(cfg["trainer.eval_interval"], target + 1, cfg["trainer.eval_interval"]))
    if cfg["trainer.eval_before_train"]:
        eval_steps.insert(0, 0)
    counts = []
    question_set = None
    for step in eval_steps:
        path = run / f"exports/aime25/dumped_evals/global_step_{step}_evals/aime25.jsonl"
        rows = parse_jsonl(read(path))
        groups = defaultdict(list)
        for row in rows:
            if not math.isfinite(row["score"]):
                raise ValueError("Nonfinite raw evaluation score")
            groups[row["input_prompt"]].append(row["score"] > 0)
        if len(groups) != questions or {len(values) for values in groups.values()} != {8}:
            raise ValueError(
                f"Step {step} evaluation is not {questions} complete eight-response groups"
            )
        if question_set is not None and set(groups) != question_set:
            raise ValueError("Evaluation questions changed between checkpoints")
        question_set = set(groups)
        correct = sum(sum(values) for values in groups.values())
        solved = sum(any(values) for values in groups.values())
        accuracy, coverage = correct / len(rows), solved / questions
        for key, value in (("eval/aime25/avg@8", accuracy), ("eval/aime25/pass@8", coverage)):
            logged = evaluations.get(step, {}).get(key)
            if logged is None or not math.isclose(logged, value, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"Step {step} raw evaluation disagrees with logged {key}")
        counts.append(
            {
                "step": step,
                "responses": len(rows),
                "questions": questions,
                "correct_responses": correct,
                "solved_questions": solved,
                "avg@8": accuracy,
                "pass@8": coverage,
            }
        )
    rank_rows = parse_jsonl(read(run / "rank-diagnostics.jsonl"))
    final = [row for row in rank_rows if row["step"] == target]
    if len(final) != 1 or not final[0]["layers"]:
        raise ValueError("Missing unique final accumulated-rank diagnostic")
    layers = list(final[0]["layers"].values())
    for layer in layers:
        if (
            not math.isfinite(layer["stable_rank"])
            or layer["stable_rank"] < 0
            or not 0 <= layer["energy_outside_first_direction"] <= 1
        ):
            raise ValueError("Invalid accumulated-rank diagnostic")
    initial = json.loads(read(run / "initial-adapter-audit.json"))
    if initial.get("optimizer") != "AdamW" or initial.get("betas") != [0.9, 0.999]:
        raise ValueError("Recorded startup optimizer differs from native AdamW with matched betas")
    maximum_allocated = max(memory, key=lambda row: row["peak_allocated_bytes"])
    maximum_reserved = max(memory, key=lambda row: row["peak_reserved_bytes"])
    windows = []
    for start in range(1, target + 1, 20):
        end = min(start + 19, target)
        windows.append(
            {
                "start": start,
                "end": end,
                "correctness": statistics.mean(
                    training[step]["reward/mean_positive_reward"] for step in range(start, end + 1)
                ),
            }
        )
    return {
        "run_id": run_id,
        "protocol_audit_passed": True,
        "exit_status": 0,
        "training_updates": len(training),
        "refresh_after_updates": merge_steps,
        "training_correctness_windows": windows,
        "initial_optimizer_audit": {
            key: initial.get(key)
            for key in ("optimizer", "betas", "epsilon", "trainable_parameters")
        },
        "training_allocator": {
            "ranks": cfg["trainer.placement.policy_num_gpus_per_node"],
            "records": len(memory),
            "windows": sorted({row["window"] for row in memory}),
            "max_allocated_gib": maximum_allocated["peak_allocated_bytes"] / 2**30,
            "max_reserved_gib": maximum_reserved["peak_reserved_bytes"] / 2**30,
            "max_allocated_at": {key: maximum_allocated[key] for key in ("step", "rank")},
            "max_reserved_at": {key: maximum_reserved[key] for key in ("step", "rank")},
        },
        "evaluations": counts,
        "final_rank": {
            "layers": len(layers),
            "mean_stable_rank": statistics.mean(layer["stable_rank"] for layer in layers),
            "mean_energy_outside_first_direction": statistics.mean(
                layer["energy_outside_first_direction"] for layer in layers
            ),
        },
        "source_sha256": hashes,
        "limitations": "This verifies the recorded protocol, not performance parity or a multi-seed result. Training allocator peaks cover the worker's declared window, not total GPU memory, inference, weight synchronization, export, or phone memory. Initial optimizer/parameter metadata comes from the recorded startup audit. No weight files are read or removed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = audit_run(root, args.run_id)
    output = root / f"research/data/{args.run_id}-completion-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "protocol_audit_passed": True,
                "training_updates": report["training_updates"],
                "training_allocator": report["training_allocator"],
                "final_evaluation": report["evaluations"][-1],
            }
        )
    )


if __name__ == "__main__":
    main()
