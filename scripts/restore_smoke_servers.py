#!/usr/bin/env python3
"""Restart selected inference services after a training run frees their GPUs."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def gpu_memory_mib():
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    return [int(line.split()[0]) for line in lines]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="qwen3-4b-base-reinforce-relora-smoke20-20260922-01")
    parser.add_argument("--port", type=int, action="append")
    args = parser.parse_args()
    run = (Path("runs") / args.run_id).resolve()
    if run.parent != Path("runs").resolve():
        parser.error("run-id must be a single directory name")
    status = run.parent / "logs" / f"{run.name}.exit-status"
    deadline = time.monotonic() + 24 * 3600
    while not status.exists() and time.monotonic() < deadline:
        time.sleep(10)
    if not status.exists():
        (run / "restore-status.json").write_text(json.dumps({"status": "run timeout"}) + "\n")
        return

    source = json.loads(Path("/tmp/condrl-smoke20-existing-servers.json").read_text())
    if args.port:
        source = [
            server
            for server in source
            if int(server["command"][server["command"].index("--port") + 1]) in args.port
        ]
        if len(source) != len(set(args.port)):
            parser.error("requested port not present in saved server commands")
    gpus = [
        int(server["command"][server["command"].index("--port") + 1]) - 18000 for server in source
    ]
    # SkyRL tears down Ray and vLLM after writing exit status.
    clear_deadline = time.monotonic() + 20 * 60
    while time.monotonic() < clear_deadline:
        memory = gpu_memory_mib()
        if all(memory[gpu] < 8000 for gpu in gpus):
            break
        time.sleep(10)
    else:
        (run / "restore-status.json").write_text(
            json.dumps({"status": "GPU memory did not clear; servers not restarted"}) + "\n"
        )
        return

    api_pid = 4036450
    api_env_path = Path(f"/proc/{api_pid}/environ")
    if not api_env_path.exists():
        (run / "restore-status.json").write_text(
            json.dumps({"status": "original API server environment unavailable"}) + "\n"
        )
        return
    common_env = {
        name.decode(): value.decode()
        for entry in api_env_path.read_bytes().split(b"\0")
        if b"=" in entry
        for name, value in [entry.split(b"=", 1)]
    }
    compat = "/usr/local/cuda-13.0/compat"
    library_path = common_env.get("LD_LIBRARY_PATH", "")
    common_env["LD_LIBRARY_PATH"] = compat + (f":{library_path}" if library_path else "")
    common_env["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libstdc++.so.6"

    restarted = []
    for server in source:
        args = server["command"]
        port = int(args[args.index("--port") + 1])
        gpu = port - 18000
        if not 0 <= gpu < 8:
            raise ValueError(f"Unexpected inference server port: {port}")
        env = dict(common_env, CUDA_VISIBLE_DEVICES=str(gpu))
        with (run / f"restored-server-{port}.log").open("w") as out:
            process = subprocess.Popen(
                args,
                cwd=server["cwd"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        restarted.append({"port": port, "gpu": gpu, "pid": process.pid})
    (run / "restore-status.json").write_text(
        json.dumps(
            {
                "status": "started",
                "time": datetime.now(timezone.utc).isoformat(),
                "servers": restarted,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
