"""Run the matched restart-ramp/control pair sequentially and retain measurements."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from train import PYTHON, ROOT, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Unique date/suffix, e.g. 20261001-01")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in {".", ".."}:
        parser.error("tag must be a directory-name suffix")
    for warmup in (5, 0):
        run_id = f"qwen3-4b-base-grpo-relora-r1-warmup{warmup}-{args.tag}"
        subprocess.run(
            [
                str(PYTHON),
                str(ROOT / "scripts/train.py"),
                "--mode",
                "relora",
                "--config",
                f"qwen3-4b-base-grpo-relora-r1-warmup{warmup}.json",
                "--gpus",
                "0,1,2,3,4,5,6,7",
                "--run-id",
                run_id,
                "--launch",
            ],
            check=True,
            env=environment("0,1,2,3,4,5,6,7"),
        )
        run = ROOT / "runs" / run_id
        exitfile = ROOT / "runs/logs" / f"{run_id}.exit-status"
        while not (run / "config.json").exists():
            if exitfile.exists():
                raise RuntimeError(f"{run_id} failed before initialization")
            time.sleep(2)
        with (run / "vram-monitor.log").open("w") as output:
            monitor = subprocess.Popen(
                [
                    str(PYTHON),
                    str(ROOT / "scripts/monitor_vram.py"),
                    "--run-id",
                    run_id,
                    "--interval",
                    "10",
                ],
                env=environment("0,1,2,3,4,5,6,7"),
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        (run / "vram-monitor.pid").write_text(f"{monitor.pid}\n")
        while not exitfile.exists():
            time.sleep(60)
        status = int(exitfile.read_text())
        monitor.wait(timeout=60)
        if status:
            raise RuntimeError(
                f"{run_id} exited {status}; control not started, diagnostics retained"
            )
        removed = []
        for path in (run / "checkpoints", run / "exports", run / "relora-rank-factors.pt"):
            if not path.exists():
                continue
            size = (
                sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                if path.is_dir()
                else path.stat().st_size
            )
            removed.append({"path": str(path.relative_to(run)), "bytes": size})
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        (run / "checkpoint-cleanup.json").write_text(
            json.dumps({"removed": removed, "measurements_retained": True}, indent=2) + "\n"
        )
        print(json.dumps({"run_id": run_id, "exit_status": status, "cleanup": removed}), flush=True)


if __name__ == "__main__":
    main()
