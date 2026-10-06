"""Gate three shared-step-40 branches on a native one-update validation.

Runs sequentially on all eight GPUs. Stop on any failed gate/run; never fall
back to another checkpoint, auto-delete results, or start a competing job.
"""

import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from train import PYTHON, ROOT, environment

ORIGIN = "qwen3-4b-base-grpo-shared-prefix-r1-20261005-01"
GPUS = "0,1,2,3,4,5,6,7"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_native_validation(run):
    audit = rows(run / "boundary-reset-audit.jsonl")
    if len(audit) != 1:
        raise ValueError("Expected one native refresh boundary")
    boundary = audit[0]
    if (
        boundary["step"] != 40
        or boundary["scheduler_step_after_boundary"] != 40
        or boundary["optimizer_state_entries_after_boundary"] != 0
        or boundary["branch"] != "guided"
        or not any(not row["fallback_to_random"] for row in boundary["layers"])
    ):
        raise ValueError("Native guided-reset state validation failed")
    metrics = rows(run / "metrics.jsonl")
    if len(metrics) != 1 or metrics[0]["step"] != 41:
        raise ValueError("Validation must execute exactly update 41")
    norm = metrics[0]["metrics"]["policy/grad_norm"]
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Native post-reset update has an invalid gradient")
    syncs = rows(run / "backbone-sync-audit.jsonl")
    if not any(row["step"] == 41 and row["base_then_adapter"] for row in syncs):
        raise ValueError("Merged backbone and fresh adapter were not synchronized")
    for rank in range(8):
        memory = rows(run / f"training-memory-rank{rank}.jsonl")
        if len(memory) != 1 or memory[0]["step"] != 41:
            raise ValueError("Missing native all-rank memory validation")
    return {
        "post_reset_grad_norm": norm,
        "guided_layers": sum(not row["fallback_to_random"] for row in boundary["layers"]),
        "native_boundary_metrics": boundary["boundary_metrics"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20261006-01")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in {".", ".."}:
        parser.error("Tag must be a directory-name component")
    (ROOT / ".tmp").mkdir(exist_ok=True)
    lock = (ROOT / ".tmp/boundary-comparison.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = ROOT / "runs" / ORIGIN / "comparison-stage.json"

    def status(stage, **fields):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "status": stage, **fields}
        status_path.write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    try:
        checkpoint = ROOT / "runs" / ORIGIN / "checkpoints/global_step_40/trainer_state.pt"
        if not checkpoint.is_file():
            raise ValueError("Shared pre-merge checkpoint missing")
        precision = ROOT / "research/data" / f"{ORIGIN}-local-merge-precision-step40.json"
        report = json.loads(precision.read_text())
        if (
            report["checkpoint_global_step"] != 40
            or report["verification"]["verified_adapter_tensors"] != 504
        ):
            raise ValueError("Verified pre-merge numerical check missing")
        configs = {
            branch: json.loads(
                (ROOT / "configs" / f"qwen3-4b-base-grpo-boundary-{branch}-r1.json").read_text()
            )
            for branch in ("standard", "random", "guided")
        }
        reference = {
            k: v
            for k, v in configs["standard"].items()
            if k not in {"trainer.boundary_branch", "trainer.relora_enable_merge"}
        }
        for cfg in configs.values():
            common = {
                k: v
                for k, v in cfg.items()
                if k not in {"trainer.boundary_branch", "trainer.relora_enable_merge"}
            }
            if common != reference or cfg["trainer.resume_path"] != str(
                checkpoint.parent.relative_to(ROOT)
            ):
                raise ValueError("Branch recipe or checkpoint mismatch")
        validation = {
            **configs["guided"],
            "trainer.boundary_validation_only": True,
            "trainer.max_training_steps": 41,
            "trainer.eval_interval": -1,
            "trainer.ckpt_interval": -1,
        }
        validation_path = ROOT / ".tmp" / f"boundary-validation-{args.tag}.json"
        validation_path.write_text(json.dumps(validation, indent=2) + "\n")
        plan = [("validation", str(validation_path))] + [
            (branch, f"qwen3-4b-base-grpo-boundary-{branch}-r1.json") for branch in configs
        ]
        status(
            "validated_plan",
            origin=ORIGIN,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            order=[branch for branch, _ in plan],
        )
        for branch, profile in plan:
            run_id = f"qwen3-4b-base-grpo-boundary-{branch}-r1-{args.tag}"
            run = ROOT / "runs" / run_id
            if run.exists():
                raise ValueError(f"Refusing to overwrite existing run: {run_id}")
            status("waiting_for_free_gpus", branch=branch, run_id=run_id)
            while subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip():
                time.sleep(60)
            command = [
                str(PYTHON),
                str(ROOT / "scripts/train.py"),
                "--mode",
                "guided-reset",
                "--config",
                profile,
                "--gpus",
                GPUS,
                "--run-id",
                run_id,
                "--launch",
            ]
            launched = json.loads(
                subprocess.check_output(command, env=environment(GPUS), cwd=ROOT, text=True)
            )
            status(
                "running_" + branch,
                run_id=run_id,
                launcher_pid=launched["pid"],
                target_global_step=41 if branch == "validation" else 100,
            )
            while not run.is_dir():
                if not Path(f"/proc/{launched['pid']}").exists():
                    raise RuntimeError("Launcher exited before run initialization")
                time.sleep(1)
            with (run / "monitor_vram.log").open("w") as log:
                monitor = subprocess.Popen(
                    [
                        str(PYTHON),
                        str(ROOT / "scripts/monitor_vram.py"),
                        "--run-id",
                        run_id,
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
            exit_path = ROOT / "runs/logs" / f"{run_id}.exit-status"
            while not exit_path.exists():
                if not Path(f"/proc/{launched['pid']}").exists():
                    raise RuntimeError("Training launcher vanished without an exit status")
                time.sleep(60)
            if int(exit_path.read_text()) != 0:
                raise RuntimeError(
                    f"Training failed: {run_id}, exit={exit_path.read_text().strip()}"
                )
            if branch == "validation":
                result = check_native_validation(run)
                status("native_validation_passed", run_id=run_id, **result)
            else:
                logged = rows(run / "metrics.jsonl")
                if [row["step"] for row in logged] != list(range(41, 101)):
                    raise ValueError("Continuation did not execute exactly updates 41..100")
                status("completed_" + branch, run_id=run_id, last_global_step=100)
        status("all_three_branches_completed", origin=ORIGIN)
    except Exception as error:
        status("stopped_on_failure", error=str(error))
        raise


if __name__ == "__main__":
    main()
