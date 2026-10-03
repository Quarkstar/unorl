"""Analyze matched continuation windows and paired question-level uncertainty."""

import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "research"
IDS = {
    "control": "qwen3-4b-base-grpo-standard-r1-continue-20261003-01",
    "candidate": "qwen3-4b-base-grpo-relora-refresh-r1-continue-20261003-01",
}
KEYS = (
    "reward/mean_positive_reward",
    "policy/policy_entropy",
    "policy/grad_norm",
    "generate/avg_num_tokens",
    "updates/effective_delta_l2",
    "updates/cosine_with_previous_delta",
)


def paired_questions(control, candidate, metric):
    """Require complete eight-sample evaluations of exactly the same questions."""
    left = {row["prompt_sha256"]: row for row in control["questions"]}
    right = {row["prompt_sha256"]: row for row in candidate["questions"]}
    if not left or left.keys() != right.keys():
        raise ValueError("Paired evaluation requires identical nonempty question sets")
    if metric not in {"avg@8", "pass@8"}:
        raise ValueError("Unknown evaluation metric")
    values = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["samples"] != 8 or b["samples"] != 8:
            raise ValueError("Each question must have eight completed scoring records")
        if not (0 <= a["correct"] <= 8 and 0 <= b["correct"] <= 8):
            raise ValueError("Invalid correctness count")
        value = (
            (b["correct"] - a["correct"]) / 8
            if metric == "avg@8"
            else float(b["correct"] > 0) - float(a["correct"] > 0)
        )
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def bootstrap_difference(values, seed=42, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires nonempty finite per-question differences")
    rng = np.random.default_rng(seed)
    resamples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "candidate_minus_control": float(values.mean()),
        "question_bootstrap_95_interval": np.quantile(resamples, [0.025, 0.975]).tolist(),
        "questions": len(values),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def analyze(control, candidate):
    if candidate:
        allowed = {
            "trainer.run_name",
            "trainer.ckpt_path",
            "trainer.export_path",
            "trainer.log_path",
            "trainer.relora_enable_merge",
            "trainer.relora_refresh_angle_degrees",
        }
        left, right = control["config"], candidate["config"]
        mismatch = [
            key
            for key in left.keys() | right.keys()
            if key not in allowed and left.get(key) != right.get(key)
        ]
        if mismatch:
            raise ValueError(f"Matched recipe differs: {sorted(mismatch)}")
    runs = {"control": control, "candidate": candidate}
    records = {
        name: {row["step"]: row["metrics"] for row in run["metrics"]} if run else {}
        for name, run in runs.items()
    }
    report = {
        "status": "paired branches available" if candidate else "candidate not started",
        "windows": {},
        "evaluations": {},
        "matched_recipe_verified": candidate is not None,
        "limitations": "Question bootstrap treats questions as sampling units and retains response grouping. It does not measure training-seed uncertainty, prove equivalence, or establish superiority from a single pair. Training reward windows use changing on-policy samples, not fixed test questions. No confidence interval is assigned to serial training rewards.",
    }
    for lo in (101, 121, 141, 161, 181):
        hi = lo + 19
        common = sorted(
            set(records["control"]) & set(records["candidate"]) & set(range(lo, hi + 1))
        )
        window = {"paired_updates": len(common), "complete": len(common) == 20, "metrics": {}}
        for key in KEYS:
            eligible = [s for s in common if all(key in records[n][s] for n in runs)]
            if eligible:
                means = {n: statistics.mean(records[n][s][key] for s in eligible) for n in runs}
                window["metrics"][key] = {
                    **means,
                    "candidate_minus_control": means["candidate"] - means["control"],
                    "paired_updates": len(eligible),
                }
        report["windows"][f"{lo}-{hi}"] = window
    evaluations = {
        name: {
            r["step"]: r
            for r in run.get("evaluation_question_scores", [])
            if r["benchmark"] == "aime25"
        }
        if run
        else {}
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
            if step == 100:
                initial[metric] = differences
            elif metric in initial:
                start_left = {q["prompt_sha256"] for q in evaluations["control"][100]["questions"]}
                end_left = {q["prompt_sha256"] for q in evaluations["control"][step]["questions"]}
                if start_left != end_left:
                    raise ValueError(
                        "Improvement comparison requires the same questions at both steps"
                    )
                result["difference_in_improvement_from_step100"] = bootstrap_difference(
                    differences - initial[metric]
                )
            report["evaluations"][str(step)][metric] = result
    return report


def main():
    runs = {}
    for name, rid in IDS.items():
        path = ROOT / "data" / f"{rid}.json"
        runs[name] = json.loads(path.read_text()) if path.exists() else None
    if not runs["control"]:
        raise FileNotFoundError("The continuation control snapshot is not available")
    result = analyze(runs["control"], runs["candidate"])
    (ROOT / "data/refresh-comparison-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
