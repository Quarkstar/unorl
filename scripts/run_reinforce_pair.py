#!/usr/bin/env python3
"""Run raw and batch-normalized REINFORCE sequentially under identical settings."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from train import CODE, ROOT, environment, PYTHON


CONFIGS = (
    ("reinforce", "qwen3-4b-base-reinforce-adamw-r1.json"),
    ("reinforce-batchnorm", "qwen3-4b-base-reinforce-batchnorm-adamw-r1.json"),
)


def final_step(run_id):
    metrics_path = ROOT / "runs" / run_id / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    steps = []
    for line in metrics_path.read_text().splitlines():
        try:
            steps.append(int(json.loads(line)["step"]))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return max(steps, default=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--pair-id", required=True)
    args = parser.parse_args()

    logs = ROOT / "runs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = logs / f"{args.pair_id}.status.json"
    status = {"pair_id": args.pair_id, "state": "running", "runs": []}
    status_path.write_text(json.dumps(status, indent=2) + "\n")

    for name, config in CONFIGS:
        run_id = f"qwen3-4b-base-{name}-adamw-r1-b256-20260925-01"
        command = [
            str(PYTHON),
            str(CODE / "scripts" / "train.py"),
            "--mode",
            "reinforce",
            "--run-id",
            run_id,
            "--config",
            config,
            "--gpus",
            args.gpus,
        ]
        log_path = logs / f"{run_id}.log"
        with log_path.open("x") as output:
            output.write(f"Started {datetime.now(timezone.utc).isoformat()}\n")
            output.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment(args.gpus),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (logs / f"{run_id}.pid").write_text(str(process.pid) + "\n")
            run_status = {
                "run_id": run_id,
                "pid": process.pid,
                "config": config,
                "state": "running",
            }
            status["runs"].append(run_status)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            exit_code = process.wait()

        completed_step = final_step(run_id)
        run_status.update(exit_code=exit_code, final_step=completed_step)
        if exit_code != 0 or completed_step != 100:
            run_status["state"] = "failed"
            status.update(state="failed", failed_run=run_id)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            raise SystemExit(
                f"{run_id} ended with exit={exit_code}, step={completed_step}; "
                "not starting next run"
            )

        run_status["state"] = "completed"
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    status.update(state="completed", completed_at=datetime.now(timezone.utc).isoformat())
    status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
