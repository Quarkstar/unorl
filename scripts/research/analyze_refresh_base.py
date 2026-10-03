"""Compare the original 100-step LoRA reference with gradual refresh from base."""

import argparse
import json
import statistics
from pathlib import Path

from scripts.research.analyze_refresh import KEYS, bootstrap_difference, paired_questions

ROOT = Path(__file__).resolve().parents[2] / "research"
IDS = {
    "control": "qwen3-4b-base-grpo-lora-r1-blog-20260923-01",
    "candidate": "qwen3-4b-base-grpo-relora-refresh-r1-20261003-01",
}
WINDOWS = [(1, 20), (21, 40), (41, 45), (46, 60), (61, 80), (81, 90), (91, 100), (81, 100)]


def analyze(control, candidate):
    """Retain partial-window counts and question grouping; expose recipe differences."""
    runs = {"control": control, "candidate": candidate}
    rows = {name: {r["step"]: r["metrics"] for r in run["metrics"]} for name, run in runs.items()}
    report = {
        "run_ids": {name: run["run_id"] for name, run in runs.items()},
        "completed_updates": {name: max(records, default=0) for name, records in rows.items()},
        "config_differences": {
            key: {name: run["config"].get(key) for name, run in runs.items()}
            for key in sorted(control["config"].keys() | candidate["config"].keys())
            if control["config"].get(key) != candidate["config"].get(key)
        },
        "windows": {},
        "evaluations": {},
        "limitations": (
            "One control/candidate pair, not a multi-seed replication. Recipe differences "
            "are exposed explicitly. On-policy training windows contain different trajectories and "
            "are descriptive; no independent-sample confidence interval is assigned to them. "
            "Question bootstrap preserves eight-response groups but does not measure training-seed "
            "uncertainty or establish equivalence. Completion counts do not prove process success."
        ),
    }
    for lo, hi in WINDOWS:
        common = sorted(rows["control"].keys() & rows["candidate"].keys() & set(range(lo, hi + 1)))
        window = {"updates": len(common), "complete": len(common) == hi - lo + 1, "metrics": {}}
        for key in KEYS:
            eligible = [step for step in common if all(key in rows[name][step] for name in runs)]
            if eligible:
                means = {
                    name: statistics.mean(rows[name][step][key] for step in eligible)
                    for name in runs
                }
                window["metrics"][key] = {
                    **means,
                    "candidate_minus_control": means["candidate"] - means["control"],
                    "updates": len(eligible),
                }
        report["windows"][f"{lo}-{hi}"] = window
    evaluations = {
        name: {
            r["step"]: r for r in run["evaluation_question_scores"] if r["benchmark"] == "aime25"
        }
        for name, run in runs.items()
    }
    initial = {}
    for step in sorted(evaluations["control"].keys() & evaluations["candidate"].keys()):
        report["evaluations"][str(step)] = {}
        for metric in ("avg@8", "pass@8"):
            differences = paired_questions(
                evaluations["control"][step], evaluations["candidate"][step], metric
            )
            result = bootstrap_difference(differences)
            if step == 0:
                initial[metric] = differences
            elif metric in initial:
                start = {q["prompt_sha256"] for q in evaluations["control"][0]["questions"]}
                end = {q["prompt_sha256"] for q in evaluations["control"][step]["questions"]}
                if start != end:
                    raise ValueError(
                        "Improvement comparison requires identical questions at both steps"
                    )
                result["difference_in_improvement_from_step0"] = bootstrap_difference(
                    differences - initial[metric]
                )
            report["evaluations"][str(step)][metric] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", default=IDS["control"])
    parser.add_argument("--candidate-run", default=IDS["candidate"])
    parser.add_argument("--output", default="refresh-base-comparison-analysis.json")
    args = parser.parse_args()
    ids = {"control": args.control_run, "candidate": args.candidate_run}
    runs = {
        name: json.loads((ROOT / "data" / f"{rid}.json").read_text()) for name, rid in ids.items()
    }
    report = analyze(runs["control"], runs["candidate"])
    (ROOT / "data" / args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "completed_updates": report["completed_updates"],
                "evaluation_steps": list(report["evaluations"]),
            }
        )
    )


if __name__ == "__main__":
    main()
