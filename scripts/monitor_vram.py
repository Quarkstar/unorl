"""Sample NVML memory without modifying or restarting a running experiment."""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

PHASES = {
    "eval": "evaluation",
    "generate": "rollout",
    "fwd_logprobs_values_reward": "training_forward",
    "train_critic_and_policy": "training_update",
    "sync_weights": "weight_sync",
    "save_checkpoints": "checkpoint",
    "save_hf_model": "checkpoint",
}
EVENT = re.compile(r"(Started|Finished): '([^']+)'")


class PhaseTracker:
    """Ignore nested timers that would otherwise obscure the training phase."""

    def __init__(self):
        self.active = None
        self.step = 0

    def update(self, text):
        for action, name in EVENT.findall(text):
            if action == "Started" and name == "step":
                self.step += 1
            if name not in PHASES:
                continue
            if action == "Started":
                self.active = name
            elif self.active == name:
                self.active = None

    @property
    def phase(self):
        return PHASES.get(self.active, "startup_or_between_phases")


def main():
    import pynvml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument(
        "--resume", action="store_true", help="Append samples and retain prior peaks"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a directory name")
    if args.interval <= 0:
        parser.error("interval must be positive")
    run = root / "runs" / args.run_id
    log = root / "runs/logs" / f"{args.run_id}.log"
    exitfile = log.with_suffix(".exit-status")
    pynvml.nvmlInit()
    handles = {int(i): pynvml.nvmlDeviceGetHandleByIndex(int(i)) for i in args.gpus.split(",")}
    tracker = PhaseTracker()
    offset = 0
    peaks = {}
    started = datetime.now(timezone.utc).isoformat()
    samples = 0
    intervals = [args.interval]
    if args.resume:
        prior = json.loads((run / "vram-summary.json").read_text())
        started = prior["started_utc"]
        samples = prior["samples"]
        peaks = prior["peak_used_bytes"]
        intervals = sorted(
            set(prior.get("sampling_intervals_seconds", [prior["interval_seconds"]]) + intervals)
        )
    try:
        with (run / "vram-samples.jsonl").open("a" if args.resume else "x", buffering=1) as output:
            while True:
                with log.open() as stream:
                    stream.seek(offset)
                    tracker.update(stream.read())
                    offset = stream.tell()
                readings = []
                for gpu, handle in handles.items():
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    processes = []
                    for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        try:
                            command = (
                                Path(f"/proc/{process.pid}/cmdline")
                                .read_bytes()
                                .replace(b"\0", b" ")
                                .decode(errors="replace")
                                .strip()
                            )
                        except OSError:
                            command = "process exited"
                        processes.append(
                            dict(pid=process.pid, used_bytes=process.usedGpuMemory, command=command)
                        )
                    readings.append(dict(gpu=gpu, used_bytes=info.used, processes=processes))
                    for phase in ("all_observed_phases", tracker.phase):
                        key = f"{phase}/gpu{gpu}"
                        peaks[key] = max(peaks.get(key, 0), info.used)
                output.write(
                    json.dumps(
                        dict(
                            utc=datetime.now(timezone.utc).isoformat(),
                            step=tracker.step,
                            phase=tracker.phase,
                            readings=readings,
                        )
                    )
                    + "\n"
                )
                samples += 1
                summary = {
                    "started_utc": started,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "interval_seconds": args.interval,
                    "sampling_intervals_seconds": intervals,
                    "samples": samples,
                    "finished": exitfile.exists(),
                    "peak_used_bytes": peaks,
                    "measurement": "NVML sampled device memory, including all resident processes",
                    "limitations": "May miss sub-interval peaks. Phases follow buffered console timers. Not PyTorch allocator high-water marks. Startup before monitor attachment is not covered.",
                }
                temporary = run / "vram-summary.tmp"
                temporary.write_text(json.dumps(summary, indent=2) + "\n")
                temporary.replace(run / "vram-summary.json")
                if exitfile.exists():
                    break
                time.sleep(args.interval)
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
