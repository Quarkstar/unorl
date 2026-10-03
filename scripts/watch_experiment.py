"""Verify a live launcher and record hourly experiment health and book snapshots."""

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect_status(root, run_id):
    run = root / "runs" / run_id
    logroot = root / "runs/logs"
    pidfile = logroot / f"{run_id}.pid"
    exitfile = logroot / f"{run_id}.exit-status"
    rows = []
    if (run / "metrics.jsonl").exists():
        for line in (run / "metrics.jsonl").read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A writer can be midway through appending the latest row.
                continue
    last = rows[-1] if rows else {"step": None, "metrics": {}}
    pid = int(pidfile.read_text()) if pidfile.exists() else None
    alive = False
    start_ticks = None
    if pid is not None:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            alive = run_id.encode() in cmdline and fields[0] != "Z"
            start_ticks = fields[19]
        except (FileNotFoundError, ProcessLookupError):
            pass
    maximum = json.loads((run / "config.json").read_text())["trainer.max_training_steps"]
    exitcode = int(exitfile.read_text()) if exitfile.exists() else None
    if exitcode is not None:
        status = "complete" if exitcode == 0 and last["step"] == maximum else "failed_or_early_exit"
    else:
        status = "running" if alive else "launcher_missing"
    rewards = [
        row["metrics"]["reward/mean_positive_reward"]
        for row in rows
        if "reward/mean_positive_reward" in row["metrics"]
    ][-10:]
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": status,
        "pid": pid,
        "verified_alive": alive,
        "process_start_ticks": start_ticks,
        "exit_status": exitcode,
        "global_step": last["step"],
        "target_global_step": maximum,
        "last10_mean_correctness": statistics.mean(rewards) if rewards else None,
        "entropy": last["metrics"].get("policy/policy_entropy"),
        "grad_norm": last["metrics"].get("policy/grad_norm"),
        "response_tokens": last["metrics"].get("generate/avg_num_tokens"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--interval", type=float, default=3600)
    parser.add_argument("--update-book", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."} or args.interval <= 0:
        parser.error("Provide a valid run name and positive interval")
    run = ROOT / "runs" / args.run_id
    next_check = 0.0
    while True:
        status = collect_status(ROOT, args.run_id)
        terminal = status["status"] != "running"
        if terminal or time.monotonic() >= next_check:
            with (run / "hourly-status.jsonl").open("a") as output:
                output.write(json.dumps(status) + "\n")
            (run / "latest-health.json").write_text(json.dumps(status, indent=2) + "\n")
            print(json.dumps(status), flush=True)
            if args.update_book:
                python = ROOT / ".venv-docs/bin/python"
                for script in ("snapshot.py", "build.py"):
                    result = subprocess.run(
                        [str(python), str(ROOT / "scripts/research" / script)],
                        cwd=ROOT,
                        timeout=180,
                        check=False,
                    )
                    if result.returncode:
                        print(f"Book update failed: {script}, exit {result.returncode}", flush=True)
                        break
            next_check = time.monotonic() + args.interval
        if terminal:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
