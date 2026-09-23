#!/usr/bin/env python3
"""Summarize a smoke run and optionally wait for its launcher to finish."""

import argparse
import json
import time
from pathlib import Path
from statistics import mean

METRICS = {
    "accuracy": "reward/mean_positive_reward",
    "signed_reward": "reward/avg_raw_reward",
    "entropy": "policy/policy_entropy",
    "response_tokens": "generate/avg_num_tokens",
    "grad_norm": "policy/grad_norm",
    "step_seconds": "timing/step",
    "optimizer_state_entries": "optimizer/state_entries",
    "merge_rounding_error": "relora/rounding_relative_l2",
}


def read_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(run):
    records = [
        r
        for r in read_records(run / "metrics.jsonl")
        if "policy/policy_loss" in r["metrics"] or "policy/loss" in r["metrics"]
    ]
    series = {
        name: [(r["step"], float(r["metrics"][key])) for r in records if key in r["metrics"]]
        for name, key in METRICS.items()
    }
    summary = {
        "run": run.name,
        "completed_steps": max((r["step"] for r in records), default=0),
        "metrics": {
            name: {
                "first_5_mean": mean(v for _, v in values[:5]),
                "last_5_mean": mean(v for _, v in values[-5:]),
                "min": min(v for _, v in values),
                "max": max(v for _, v in values),
            }
            for name, values in series.items()
            if values
        },
        "evaluations": read_records(run / "evaluation.jsonl"),
        "merges": [r for r in records if "relora/merged_layers" in r["metrics"]],
    }
    gpu_samples = read_records(run / "gpu-memory.jsonl")
    if gpu_samples:
        summary["observed_peak_gpu_memory_mib"] = max(
            gpu["memory_mib"] for sample in gpu_samples for gpu in sample["gpus"]
        )
    (run / "smoke-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if records:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("matplotlib unavailable; JSON summary saved without a plot")
            return

        fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
        for ax, name in zip(axes.flat, list(METRICS)[:6], strict=True):
            values = series[name]
            if values:
                steps, ys = zip(*values, strict=True)
                ax.plot(steps, ys, ".-", alpha=0.5, label="Per step")
                smooth = [mean(ys[max(0, i - 4) : i + 1]) for i in range(len(ys))]
                ax.plot(steps, smooth, label="Trailing 5 steps")
            for record in summary["merges"]:
                ax.axvline(record["step"], color="gray", linestyle=":")
            ax.set(title=name.replace("_", " "), xlabel="Training step")
            ax.grid(alpha=0.2)
        axes.flat[0].legend()
        fig.suptitle(run.name)
        fig.savefig(run / "smoke-curves.png", dpi=160)
        plt.close(fig)
    print(json.dumps({"run": run.name, "completed_steps": summary["completed_steps"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.wait:
        status = args.run.parent / "logs" / f"{args.run.name}.exit-status"
        deadline = time.monotonic() + 4 * 3600
        while not status.exists() and time.monotonic() < deadline:
            time.sleep(10)
    report(args.run)


if __name__ == "__main__":
    main()
