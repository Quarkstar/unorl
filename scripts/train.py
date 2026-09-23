#!/usr/bin/env python3
"""Foreground launcher; --launch starts an audited nohup process."""

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("CONDITIONAL_PROJECT_ROOT", str(CODE)))
SKYRL = ROOT.parent / "SciBuddy/third_party/SkyRL"
PIN = "0b286bacba2bb51dfe50186b6b5d6b1e0b5f5518"
PYTHON = ROOT.parent / "SciBuddy/.venv-skyrl/bin/python"


def environment(gpus="0"):
    env = dict(os.environ)
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
        CONDITIONAL_PROJECT_ROOT=str(ROOT),
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


def write_comparison(run, run_id):
    rows = []
    for member in ["conditional", "grpo"]:
        source = ROOT / "runs" / run_id.replace("pair", member) / "evaluation.jsonl"
        if not source.exists():
            continue
        for line in source.read_text().splitlines():
            record = json.loads(line)
            metrics = record["metrics"]
            for benchmark in ["math500", "amc23", "aime25", "aime26"]:
                metric = "accuracy" if benchmark == "math500" else "pass@8"
                value = metrics.get(f"eval/{benchmark}/{metric}")
                if value is None:
                    continue
                rows.append(
                    dict(
                        method=member,
                        step=record["step"],
                        benchmark=benchmark,
                        metric=metric,
                        value=value,
                        percent=None if value is None else 100 * value,
                    )
                )
    with (run / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "step", "benchmark", "metric", "value", "percent"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--mode", choices=["conditional", "positive", "grpo", "reinforce"], default="conditional"
    )
    parser.add_argument(
        "--pair", action="store_true", help="Run conditional then GRPO sequentially"
    )
    parser.add_argument("--config", default="experiment.json", help="Profile filename in configs/")
    parser.add_argument(
        "--baseline-first", action="store_true", help="Run GRPO before conditional in a pair"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--gpus", default="0", help="Comma-separated CUDA device IDs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pair and args.mode == "reinforce":
        parser.error("reinforce is a standalone experiment; omit --pair")
    actual = subprocess.check_output(
        ["git", "-C", str(SKYRL), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PIN:
        raise SystemExit(f"SkyRL pin mismatch: expected {PIN}, got {actual}")
    mode = ("pair" if args.pair else args.mode) + ("-smoke" if args.smoke else "")
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_id = args.run_id
    model_label = "qwen3-4b-base" if args.mode == "reinforce" else "qwen25-math-1.5b"
    if run_id is None:
        attempt = 1
        while (ROOT / "runs" / f"{model_label}-{mode}-{date}-{attempt:02d}").exists():
            attempt += 1
        run_id = f"{model_label}-{mode}-{date}-{attempt:02d}"
    run = ROOT / "runs" / run_id
    cfg = json.loads((CODE / "configs" / args.config).read_text())
    cfg.setdefault("trainer.policy.model.path", str(ROOT / "models/Qwen2.5-Math-1.5B-system"))
    cfg.setdefault("data.val_data", [str(ROOT / "data/eval-training-benchmarks.parquet")])
    cfg.update(
        {
            "data.train_data": [str(ROOT / "data/train-benchmark-clean.parquet")],
            "trainer.run_name": run_id,
            "trainer.ckpt_path": str(run / "checkpoints"),
            "trainer.export_path": str(run / "exports"),
            "trainer.log_path": str(run / "infra"),
        }
    )
    if args.mode == "grpo":
        cfg.setdefault("trainer.algorithm.policy_loss_type", "regular")
        cfg.setdefault("trainer.algorithm.grpo_norm_by_std", True)
        cfg.setdefault("trainer.algorithm.eps_clip_low", 0.2)
        cfg.setdefault("trainer.algorithm.eps_clip_high", 0.2)
    if args.smoke:
        # Validate the real rollout length, reward pipeline and update batch.
        # A short response cap previously hid answer-format problems.
        cfg.update(
            {
                "trainer.max_training_steps": 1,
                "trainer.policy.optimizer_config.num_warmup_steps": 0,
                "trainer.eval_before_train": True,
                "trainer.eval_interval": 1,
                "trainer.ckpt_interval": -1,
                "trainer.hf_save_interval": -1,
                "trainer.dump_data_batch": True,
            }
        )
    module = (
        "conditional_rl.low_resource_train" if args.mode == "reinforce" else "conditional_rl.train"
    )
    command = [str(PYTHON), "-m", module] + [
        f"{key}={json.dumps(value)}" for key, value in cfg.items()
    ]
    if args.dry_run:
        print(json.dumps(dict(run_id=run_id, config=cfg, command=command), indent=2))
        return
    logs = ROOT / "runs/logs"
    logs.mkdir(parents=True, exist_ok=True)
    if args.launch:
        run.mkdir(parents=True, exist_ok=False)
        launcher = [
            str(PYTHON),
            str(Path(__file__).resolve()),
            "--run-id",
            run_id,
            "--mode",
            args.mode,
            "--config",
            args.config,
            "--gpus",
            args.gpus,
        ]
        if args.baseline_first:
            launcher.append("--baseline-first")
        if args.pair:
            launcher.append("--pair")
        if args.smoke:
            launcher.append("--smoke")
        with (logs / f"{run_id}.log").open("x") as output:
            process = subprocess.Popen(
                ["nohup", *launcher],
                cwd=ROOT,
                env=environment(args.gpus),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (logs / f"{run_id}.pid").write_text(str(process.pid) + "\n")
        print(json.dumps(dict(run_id=run_id, pid=process.pid, log=str(logs / f"{run_id}.log"))))
        return
    run.mkdir(parents=True, exist_ok=True)
    if args.pair:
        import shutil

        for relative in ["conditional_rl", "configs", "scripts"]:
            shutil.copytree(
                CODE / relative,
                run / "source" / relative,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        for member in ["grpo", "conditional"] if args.baseline_first else ["conditional", "grpo"]:
            child_id = run_id.replace("pair", member)
            child = [
                str(PYTHON),
                str(run / "source/scripts/train.py"),
                "--mode",
                member,
                "--run-id",
                child_id,
                "--config",
                args.config,
            ]
            if args.smoke:
                child.append("--smoke")
            with (logs / f"{child_id}.log").open("x") as output:
                child.extend(["--gpus", args.gpus])
                process = subprocess.Popen(
                    child,
                    cwd=ROOT,
                    env=environment(args.gpus),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                (logs / f"{child_id}.pid").write_text(str(process.pid) + "\n")
                print(f"Started {member}: {child_id}, PID {process.pid}", flush=True)
                status = process.wait()
            if status:
                (logs / f"{run_id}.exit-status").write_text(str(status) + "\n")
                raise SystemExit(status)
        write_comparison(run, run_id)
        (logs / f"{run_id}.exit-status").write_text("0\n")
        return
    (run / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    assets = json.loads((ROOT / "assets.json").read_text())
    selected_model = Path(cfg["trainer.policy.model.path"])
    if (selected_model / "download-audit.json").exists():
        assets["model"] = json.loads((selected_model / "download-audit.json").read_text())
        assets.pop("conditional_model", None)
    assets["active_train_data"] = cfg["data.train_data"]
    assets["active_eval_data"] = cfg["data.val_data"]
    (run / "assets.json").write_text(json.dumps(assets, indent=2) + "\n")
    (run / "training-data-audit.json").write_text(
        (ROOT / "data/training-data-audit.json").read_text()
    )
    model_path = Path(cfg["trainer.policy.model.path"])
    if model_path.name == "Qwen2.5-Math-1.5B-system":
        (run / "before-results.json").write_text(
            (ROOT / "runs/system-benchmarks-20260915-054613/results.json").read_text()
        )
    if (model_path / "download-audit.json").exists():
        (run / "model-assets.json").write_text((model_path / "download-audit.json").read_text())
    (run / "skyrl-commit.txt").write_text(PIN + "\n")
    # Preserve exactly the small extension used for this run.
    import shutil

    for relative in ["conditional_rl", "configs", "scripts"]:
        shutil.copytree(
            CODE / relative,
            run / "source" / relative,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    env = environment(args.gpus)
    env["SKYRL_CONDITIONAL_MODE"] = args.mode
    result = subprocess.run(command, cwd=ROOT, env=env)
    status = result.returncode if result.returncode >= 0 else 128 - result.returncode
    (logs / f"{run_id}.exit-status").write_text(str(status) + "\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
