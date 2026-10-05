"""Wait for the shared step-40 prefix, then run its read-only precision audit.

This stage never launches a training branch or removes its origin checkpoint.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from train import PYTHON, ROOT, environment


def write_status(path, status, **fields):
    row = {"utc": datetime.now(timezone.utc).isoformat(), "status": status, **fields}
    path.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("Expected a run directory name")
    run = ROOT / "runs" / args.run_id
    cfg = json.loads((run / "config.json").read_text())
    if cfg["trainer.max_training_steps"] != 40 or cfg["trainer.relora_enable_merge"]:
        raise ValueError("Only the standard-LoRA shared 40-step prefix may trigger this audit")
    exitfile = ROOT / "runs/logs" / f"{args.run_id}.exit-status"
    status_file = run / "precision-stage.json"
    write_status(status_file, "waiting_for_shared_prefix", target_global_step=40)
    while not exitfile.exists():
        pid = int((ROOT / "runs/logs" / f"{args.run_id}.pid").read_text())
        try:
            command_line = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            alive = args.run_id.encode() in command_line
        except (FileNotFoundError, ProcessLookupError):
            alive = False
        if not alive and not exitfile.exists():
            write_status(status_file, "blocked_prefix_launcher_missing", pid=pid)
            return
        time.sleep(60)
    if int(exitfile.read_text()) != 0:
        write_status(status_file, "blocked_prefix_failed", exit_status=int(exitfile.read_text()))
        return
    checkpoint = run / "checkpoints/global_step_40"
    if not (checkpoint / "trainer_state.pt").is_file():
        write_status(status_file, "blocked_missing_checkpoint", checkpoint=str(checkpoint))
        return
    # The checkpoint stage must own all eight devices for native FSDP loading.
    # Wait if another experiment/user has acquired them after prefix teardown.
    write_status(status_file, "waiting_for_free_gpus")
    while subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        time.sleep(60)
    output = ROOT / "research/data" / f"{args.run_id}-merge-precision-step40.json"
    command = [
        str(PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(ROOT / "scripts/research/check_boundary_precision.py"),
        "--run-id",
        args.run_id,
        "--step",
        "40",
        "--output",
        str(output),
    ]
    write_status(status_file, "running_precision_audit", output=str(output))
    try:
        with (run / "precision-audit.log").open("a") as log:
            log.write(json.dumps({"attempt_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment("0,1,2,3,4,5,6,7"),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=7200,
                check=False,
            )
    except subprocess.TimeoutExpired:
        write_status(status_file, "precision_audit_timed_out")
        return
    if result.returncode or not output.is_file():
        write_status(status_file, "precision_audit_failed", exit_status=result.returncode)
        return
    report = json.loads(output.read_text())
    write_status(
        status_file,
        "precision_audit_complete_needs_interpretation",
        output=str(output),
        baseline_repeat=report["baseline_repeat"],
        merge_shift=report["merge_shift"],
    )


if __name__ == "__main__":
    main()
