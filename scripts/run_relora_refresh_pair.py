"""Fork standard LoRA and gradual refresh from the same step-100 checkpoint."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from run_relora_comparison import cleanup_artifacts
from train import PYTHON, ROOT, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in {".", ".."}:
        parser.error("tag must be a directory-name suffix")
    checkpoint = (
        ROOT / "runs/qwen3-4b-base-grpo-lora-r1-blog-20260923-01/checkpoints/global_step_100"
    )
    required = [checkpoint / name for name in ("data.pt", "trainer_state.pt")]
    required += [
        checkpoint / "policy" / f"{part}_world_size_8_rank_{rank}.pt"
        for part in ("model", "optim", "extra_state")
        for rank in range(8)
    ]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(
            "Shared checkpoint must contain data position and all eight model/Adam/RNG shards"
        )
    env = environment("0,1,2,3,4,5,6,7")
    for method in ("standard", "relora-refresh"):
        run_id = f"qwen3-4b-base-grpo-{method}-r1-continue-{args.tag}"
        subprocess.run(
            [
                str(PYTHON),
                str(ROOT / "scripts/train.py"),
                "--mode",
                "relora-refresh",
                "--config",
                f"qwen3-4b-base-grpo-{method}-r1-continue.json",
                "--gpus",
                "0,1,2,3,4,5,6,7",
                "--run-id",
                run_id,
                "--launch",
            ],
            env=env,
            check=True,
        )
        run = ROOT / "runs" / run_id
        exitfile = ROOT / "runs/logs" / f"{run_id}.exit-status"
        while not (run / "config.json").exists():
            if exitfile.exists():
                raise RuntimeError(f"{run_id} failed to initialize")
            time.sleep(2)
        monitors = []
        for script, arguments in [
            ("monitor_vram.py", ["--interval", "10"]),
            ("watch_experiment.py", ["--interval", "3600", "--update-book"]),
        ]:
            with (run / f"{Path(script).stem}.log").open("w") as log:
                process = subprocess.Popen(
                    [str(PYTHON), str(ROOT / "scripts" / script), "--run-id", run_id] + arguments,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            (run / f"{Path(script).stem}.pid").write_text(f"{process.pid}\n")
            monitors.append(process)
        while not exitfile.exists():
            if monitors[1].poll() is not None:
                raise RuntimeError(
                    f"{run_id}: hourly watcher exited before training; diagnostics retained"
                )
            time.sleep(60)
        for process in monitors:
            process.wait(timeout=240)
        if int(exitfile.read_text()) != 0:
            raise RuntimeError(
                f"{run_id} failed; queued sibling not launched, checkpoints retained for debugging"
            )
        health = json.loads((run / "latest-health.json").read_text())
        if health["status"] != "complete":
            raise RuntimeError(f"{run_id} exited without reaching global step 200")
        removed = cleanup_artifacts(run)
        (run / "checkpoint-cleanup.json").write_text(
            json.dumps({"removed": removed, "evaluations_and_diagnostics_retained": True}, indent=2)
            + "\n"
        )
        print(json.dumps({"run_id": run_id, "completed": True, "cleanup": removed}), flush=True)


if __name__ == "__main__":
    main()
