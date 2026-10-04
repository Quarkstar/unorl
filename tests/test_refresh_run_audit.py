"""Completion auditing must reject incomplete records and optimizer confounds."""

import json
from pathlib import Path

import pytest

from scripts.research.audit_refresh_run import audit_run


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


@pytest.fixture
def recorded_run(tmp_path):
    run_id = "synthetic-refresh"
    run = tmp_path / "runs" / run_id
    run.mkdir(parents=True)
    logs = tmp_path / "runs/logs"
    logs.mkdir()
    (logs / f"{run_id}.exit-status").write_text("0\n")
    profile = (
        Path(__file__).resolve().parents[1] / "configs/qwen3-4b-base-grpo-relora-refresh-r1.json"
    )
    cfg = json.loads(profile.read_text())
    cfg.update(
        {
            "trainer.max_training_steps": 4,
            "trainer.placement.policy_num_gpus_per_node": 2,
            "trainer.relora_first_merge_step": 2,
            "trainer.relora_merge_interval": 4,
            "trainer.eval_interval": 2,
        }
    )
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "initial-adapter-audit.json").write_text(
        json.dumps({"optimizer": "AdamW", "betas": [0.9, 0.999], "trainable_parameters": 24})
    )
    metric = {
        "reward/mean_positive_reward": 0.5,
        "policy/policy_entropy": 0.1,
        "policy/grad_norm": 0.02,
        "generate/avg_num_tokens": 100,
        "updates/effective_delta_l2": 0.03,
    }
    write_jsonl(run / "metrics.jsonl", [{"step": step, "metrics": metric} for step in range(1, 5)])
    for rank in range(2):
        write_jsonl(
            run / f"training-memory-rank{rank}.jsonl",
            [
                {
                    "step": step,
                    "rank": rank,
                    "peak_allocated_bytes": 1024**2,
                    "peak_reserved_bytes": 2 * 1024**2,
                    "window": "synthetic training window",
                    "merged": step == 2,
                }
                for step in range(1, 5)
            ],
        )
    write_jsonl(
        run / "evaluation.jsonl",
        [
            {"step": step, "metrics": {"eval/aime25/avg@8": 0.5, "eval/aime25/pass@8": 1.0}}
            for step in (0, 2, 4)
        ],
    )
    for step in (0, 2, 4):
        folder = run / f"exports/aime25/dumped_evals/global_step_{step}_evals"
        folder.mkdir(parents=True)
        write_jsonl(
            folder / "aime25.jsonl",
            [
                {"input_prompt": f"question {question}\u2028body", "score": int(sample < 4)}
                for question in range(2)
                for sample in range(8)
            ],
        )
    write_jsonl(
        run / "rank-diagnostics.jsonl",
        [
            {
                "step": 4,
                "layers": {
                    "projection": {"stable_rank": 1.1, "energy_outside_first_direction": 0.09}
                },
            }
        ],
    )
    return tmp_path, run_id, run


def test_complete_protocol_and_provenance(recorded_run):
    root, run_id, _ = recorded_run
    report = audit_run(root, run_id, questions=2)
    assert report["protocol_audit_passed"]
    assert report["training_updates"] == 4
    assert report["refresh_after_updates"] == [2]
    assert report["training_allocator"]["records"] == 8
    assert report["training_allocator"]["max_allocated_gib"] == 1 / 1024
    assert report["evaluations"][-1]["correct_responses"] == 8
    assert report["evaluations"][-1]["solved_questions"] == 2
    assert len(report["source_sha256"]) == 11


def test_missing_training_memory_record_is_not_completion(recorded_run):
    root, run_id, run = recorded_run
    path = run / "training-memory-rank1.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Rank 1 memory records"):
        audit_run(root, run_id, questions=2)


def test_raw_score_mismatch_is_not_trusted(recorded_run):
    root, run_id, run = recorded_run
    path = run / "exports/aime25/dumped_evals/global_step_4_evals/aime25.jsonl"
    rows = [json.loads(line) for line in path.read_text().split("\n") if line]
    rows[0]["score"] = 0
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="raw evaluation disagrees"):
        audit_run(root, run_id, questions=2)


def test_optimizer_confounds_are_rejected(recorded_run):
    root, run_id, run = recorded_run
    path = run / "initial-adapter-audit.json"
    initial = json.loads(path.read_text())
    initial["optimizer"] = "SGD"
    path.write_text(json.dumps(initial))
    with pytest.raises(ValueError, match="optimizer differs"):
        audit_run(root, run_id, questions=2)
