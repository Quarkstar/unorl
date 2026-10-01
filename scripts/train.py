#!/usr/bin/env python3
"""Foreground launcher; --launch starts an audited nohup process."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("UNORL_PROJECT_ROOT", str(CODE)))
SKYRL = Path(os.environ.get("SKYRL_ROOT", ROOT.parent / "SciBuddy/third_party/SkyRL"))
PIN = "0b286bacba2bb51dfe50186b6b5d6b1e0b5f5518"
PYTHON = Path(os.environ.get("SKYRL_PYTHON", ROOT.parent / "SciBuddy/.venv-skyrl/bin/python"))


def environment(gpus="0"):
    env = dict(os.environ)
    temp_root = ROOT / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    ray_temp_root = ROOT.parent / ".raytmp"
    ray_temp_root.mkdir(parents=True, exist_ok=True)
    for key, relative in {
        "HF_HOME": ".cache/huggingface",
        "XDG_CACHE_HOME": ".cache",
        "TORCH_HOME": ".cache/torch",
        "TORCH_EXTENSIONS_DIR": ".cache/torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": ".cache/torch_inductor",
        "VLLM_CACHE_ROOT": ".cache/vllm",
        "TRITON_CACHE_DIR": ".cache/triton",
        "CUDA_CACHE_PATH": ".cache/cuda",
    }.items():
        env[key] = str(ROOT / relative)
    env.update(
        TMPDIR=str(temp_root),
        RAY_TMPDIR=str(ray_temp_root),
        UNORL_PROJECT_ROOT=str(ROOT),
        PYTHONPATH=str(ROOT / ".deps") + ":" + str(CODE) + ":" + str(SKYRL),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        RAY_USAGE_STATS_ENABLED="0",
        OMP_NUM_THREADS="4",
        MAX_JOBS="4",
        SKYRL_LD_LIBRARY_PATH_EXPORT="1",
        CUDA_HOME="/usr/local/cuda-13.0",
    )
    env["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libstdc++.so.6"
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.0/compat" + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["CUDA_VISIBLE_DEVICES"] = gpus
    return env


def load_config(profile):
    """Resolve local asset paths without embedding a machine-specific project root."""
    cfg = json.loads((CODE / "configs" / profile).read_text())
    for key in ("trainer.policy.model.path", "trainer.critic.model.path"):
        if cfg.get(key) and not Path(cfg[key]).is_absolute():
            cfg[key] = str(ROOT / cfg[key])
    for key in ("data.train_data", "data.val_data"):
        if key in cfg:
            cfg[key] = [
                str(ROOT / value) if not Path(value).is_absolute() else value for value in cfg[key]
            ]
    cfg.setdefault("data.train_data", [str(ROOT / "data/train-benchmark-clean.parquet")])
    cfg["trainer.project_name"] = "unorl"
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["grpo", "reinforce", "ppo", "nora-merge", "relora"], default="reinforce"
    )
    parser.add_argument("--config", help="Profile in configs/; defaults depend on --mode")
    parser.add_argument("--gpus", default="0", help="Explicit comma-separated device IDs")
    parser.add_argument("--run-id")
    parser.add_argument("--launch", action="store_true", help="Run detached and log under runs/")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config without using GPUs")
    args = parser.parse_args()
    if args.config is None:
        args.config = {
            "grpo": "qwen3-4b-base-grpo-lora-r1-blog.json",
            "reinforce": "qwen3-4b-base-reinforce-batchnorm-adamw-r1.json",
            "ppo": "qwen3-4b-base-ppo-lora-r1-valuewarmup.json",
            "nora-merge": "qwen3-4b-base-grpo-nora-merge-r1.json",
            "relora": "qwen3-4b-base-grpo-relora-r1-warmup5.json",
        }[args.mode]
    run_id = args.run_id or f"unorl-{args.mode}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("run-id must be a directory name")
    run = ROOT / "runs" / run_id
    cfg = load_config(args.config)
    cfg.update(
        {
            "trainer.run_name": run_id,
            "trainer.ckpt_path": str(run / "checkpoints"),
            "trainer.export_path": str(run / "exports"),
            "trainer.log_path": str(run / "infra"),
        }
    )
    module = {
        "grpo": "unorl.train",
        "reinforce": "unorl.reinforce_adamw_train",
        "ppo": "unorl.ppo_train",
        "nora-merge": "unorl.nora_merge_train",
        "relora": "unorl.relora_train",
    }[args.mode]
    command = [str(PYTHON), "-m", module] + [f"{k}={json.dumps(v)}" for k, v in cfg.items()]
    if args.dry_run:
        print(json.dumps({"run_id": run_id, "config": cfg, "command": command}, indent=2))
        return
    actual = subprocess.check_output(
        ["git", "-C", str(SKYRL), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PIN:
        raise SystemExit(f"SkyRL pin mismatch: expected {PIN}, got {actual}")
    logs = ROOT / "runs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    if args.launch:
        if run.exists():
            raise SystemExit(f"Run already exists: {run}")
        launcher = [
            str(PYTHON),
            str(Path(__file__).resolve()),
            "--mode",
            args.mode,
            "--config",
            args.config,
            "--gpus",
            args.gpus,
            "--run-id",
            run_id,
        ]
        with (logs / f"{run_id}.log").open("x") as output:
            process = subprocess.Popen(
                launcher,
                cwd=ROOT,
                env=environment(args.gpus),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (logs / f"{run_id}.pid").write_text(f"{process.pid}\n")
        print(json.dumps({"run_id": run_id, "pid": process.pid}))
        return
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (run / "skyrl-commit.txt").write_text(PIN + "\n")
    (run / "unorl-commit.txt").write_text(
        subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True)
    )
    for source in (ROOT / "assets.json", ROOT / "data/training-data-audit.json"):
        if source.exists():
            shutil.copy2(source, run / source.name)
    model_audit = Path(cfg["trainer.policy.model.path"]) / "download-audit.json"
    if model_audit.exists():
        shutil.copy2(model_audit, run / "model-assets.json")
    for relative in ["unorl", "configs", "scripts"]:
        shutil.copytree(
            CODE / relative, run / "source" / relative, ignore=shutil.ignore_patterns("__pycache__")
        )
    result = subprocess.run(command, cwd=ROOT, env=environment(args.gpus))
    status = result.returncode if result.returncode >= 0 else 128 - result.returncode
    (logs / f"{run_id}.exit-status").write_text(f"{status}\n")
    sys.exit(status)


if __name__ == "__main__":
    main()
