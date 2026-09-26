"""Export post-step-60 truncation evidence from retained evaluation dumps."""

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "Batch-normalized REINFORCE": "qwen3-4b-base-reinforce-batchnorm-adamw-r1-b256-20260925-01",
    "Vanilla REINFORCE": "qwen3-4b-base-reinforce-adamw-r1-b256-20260925-01",
    "Rank-1 GRPO": "qwen3-4b-base-grpo-lora-r1-blog-20260923-01",
}


def main():
    evidence = []
    for label, run_id in RUNS.items():
        root = ROOT / "runs" / run_id
        result = {"label": label, "run_id": run_id, "evaluations": [], "unreadable": []}
        for step in [0, 20, 40, 60, 80, 100]:
            folder = root / f"exports/aime25/dumped_evals/global_step_{step}_evals"
            path = folder / "aime25.jsonl"
            try:
                rows = [json.loads(line) for line in path.read_text().splitlines()]
            except (FileNotFoundError, json.JSONDecodeError):
                result["unreadable"].append(step)
                continue
            if len(rows) != 240:
                raise ValueError(f"Incomplete evaluation: {path}")
            groups = defaultdict(list)
            for row in rows:
                groups[str(row["env_extras"]["extra_info"]["index"])].append(row)
            counts = [
                dict(
                    question_id=key,
                    correct=sum(row["score"] > 0 for row in group),
                    truncated=sum(row["stop_reason"] == "length" for row in group),
                )
                for key, group in groups.items()
            ]
            metrics = json.loads((folder / "aggregated_results.jsonl").read_text())
            result["evaluations"].append(
                {
                    "step": step,
                    "responses": len(rows),
                    "questions": len(groups),
                    "correct": sum(row["score"] > 0 for row in rows),
                    "truncated": sum(row["stop_reason"] == "length" for row in rows),
                    "truncated_correct": sum(
                        row["score"] > 0 and row["stop_reason"] == "length" for row in rows
                    ),
                    "solved": sum(
                        any(row["score"] > 0 for row in group) for group in groups.values()
                    ),
                    "avg_tokens": metrics["eval/all/generate/avg_num_tokens"],
                    "by_question": counts,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        result["training_windows"] = []
        for lo, hi in [(1, 20), (41, 60), (61, 80), (81, 100)]:
            ms = [row["metrics"] for row in rows if lo <= row["step"] <= hi]
            result["training_windows"].append(
                {
                    "start": lo,
                    "end": hi,
                    **{
                        key: statistics.mean(m[key] for m in ms if key in m)
                        for key in [
                            "reward/mean_positive_reward",
                            "policy/policy_entropy",
                            "policy/grad_norm",
                            "generate/avg_num_tokens",
                        ]
                    },
                }
            )
        evidence.append(result)
    (ROOT / "research/data/truncation-analysis.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
