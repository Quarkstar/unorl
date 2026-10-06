"""Start the approved update-cap trial after a deadline and comparison success."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from train import PYTHON, ROOT, environment

ORIGIN = "qwen3-4b-base-grpo-shared-prefix-r1-20261005-01"
GPUS = "0,1,2,3,4,5,6,7"
PROFILE = "qwen3-4b-base-grpo-boundary-update-cap-r1.json"


def source_hashes():
    paths = sorted((ROOT / "unorl").glob("*.py")) + [
        ROOT / "scripts/train.py",
        ROOT / "configs" / PROFILE,
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--not-before", required=True, help="ISO timestamp including UTC offset")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.not_before)
    if deadline.tzinfo is None:
        parser.error("not-before requires a timezone")
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("Invalid run ID")
    origin = ROOT / "runs" / ORIGIN
    status_path = origin / "next-update-cap-stage.json"
    lock = (ROOT / ".tmp/update-cap-queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = source_hashes()
    previous_status = None

    def status(stage, **fields):
        nonlocal previous_status
        if stage == previous_status and not fields:
            return
        previous_status = stage
        row = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "status": stage,
            "not_before": deadline.isoformat(),
            "run_id": args.run_id,
            **fields,
        }
        status_path.write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    try:
        trial = json.loads((ROOT / "configs" / PROFILE).read_text())
        reference = json.loads(
            (ROOT / "configs/qwen3-4b-base-grpo-boundary-random-r1.json").read_text()
        )
        if trial.pop("trainer.update_cap_budget_ratio") != 1.0 or trial != reference:
            raise ValueError(
                "Trial must differ from the random-reset recipe only by its update cap"
            )
        if not (ROOT / reference["trainer.resume_path"] / "trainer_state.pt").is_file():
            raise ValueError("Shared pre-merge checkpoint missing")
        plan = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "not_before": deadline.isoformat(),
            "run_id": args.run_id,
            "profile": PROFILE,
            "source_sha256": expected,
            "origin_checkpoint": reference["trainer.resume_path"],
            "end_global_step": 100,
            "wait_for_comparison": True,
            "gpus": GPUS,
            "vram_sample_seconds": 10,
        }
        (origin / "next-update-cap-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        while True:
            stage = json.loads((origin / "comparison-stage.json").read_text())["status"]
            if stage == "stopped_on_failure":
                raise RuntimeError(
                    "The current comparison failed; next run is not authorized to bypass that gate"
                )
            if datetime.now(timezone.utc) < deadline:
                status("waiting_for_one_hour_deadline")
            elif stage != "all_three_branches_completed":
                status("waiting_for_current_comparison")
            else:
                break
            time.sleep(60)
        for branch in ("standard", "random", "guided"):
            rid = f"qwen3-4b-base-grpo-boundary-{branch}-r1-20261006-01"
            if (ROOT / "runs/logs" / f"{rid}.exit-status").read_text().strip() != "0":
                raise ValueError("Comparison branch did not exit successfully")
            evals = [
                json.loads(line)
                for line in (ROOT / "runs" / rid / "evaluation.jsonl").read_text().splitlines()
            ]
            if not any(row["step"] == 100 for row in evals):
                raise ValueError("Comparison's final evaluation is missing")
        status("waiting_for_free_gpus")
        while subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip():
            time.sleep(60)
        if source_hashes() != expected:
            raise ValueError("Experiment source changed while queued; refusing an unaudited launch")
        run = ROOT / "runs" / args.run_id
        if run.exists():
            raise ValueError("Refusing to overwrite an existing run")
        launched = json.loads(
            subprocess.check_output(
                [
                    str(PYTHON),
                    str(ROOT / "scripts/train.py"),
                    "--mode",
                    "update-cap",
                    "--config",
                    PROFILE,
                    "--run-id",
                    args.run_id,
                    "--gpus",
                    GPUS,
                    "--launch",
                ],
                cwd=ROOT,
                env=environment(GPUS),
                text=True,
            )
        )
        status("running_update_cap", launcher_pid=launched["pid"], target_global_step=100)
        while not run.exists():
            if not Path(f"/proc/{launched['pid']}").exists():
                raise RuntimeError("Launcher exited before creating its run directory")
            time.sleep(1)
        with (run / "monitor_vram.log").open("w") as log:
            monitor = subprocess.Popen(
                [
                    str(PYTHON),
                    str(ROOT / "scripts/monitor_vram.py"),
                    "--run-id",
                    args.run_id,
                    "--gpus",
                    GPUS,
                    "--interval",
                    "10",
                ],
                cwd=ROOT,
                env=environment(GPUS),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        (run / "monitor_vram.pid").write_text(str(monitor.pid) + "\n")
        exit_path = ROOT / "runs/logs" / f"{args.run_id}.exit-status"
        while not exit_path.exists():
            if not Path(f"/proc/{launched['pid']}").exists():
                raise RuntimeError("Training launcher vanished without exit status")
            time.sleep(60)
        if exit_path.read_text().strip() != "0":
            raise RuntimeError("Update-cap run failed; inspect its log")
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        if [row["step"] for row in metrics] != list(range(41, 101)):
            raise ValueError("The trial did not execute exactly updates 41..100")
        status("completed_update_cap", last_global_step=100)
    except Exception as error:
        status("stopped_on_failure", error=str(error))
        raise


if __name__ == "__main__":
    main()
