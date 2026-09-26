"""Parallel HTTP download with official SHA256 verification."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="Qwen/Qwen3-4B-Base")
parser.add_argument("--output", default="models/Qwen3-4B-Base")
parser.add_argument("--revision")
args = parser.parse_args()
MODEL = (ROOT / args.output).resolve()
MODEL.mkdir(parents=True, exist_ok=True)
REPO = args.repo
REVISION = args.revision or HfApi().model_info(REPO).sha
info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
print(f"Downloading {REPO} revision {REVISION}", flush=True)
snapshot_download(
    repo_id=REPO,
    revision=REVISION,
    local_dir=MODEL,
    allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "README.md", "LICENSE*"],
    max_workers=4,
)
files = [f for f in info.siblings if f.rfilename.endswith(".safetensors")]
manifest = []
for f in files:
    manifest.extend(
        [
            f"https://huggingface.co/{REPO}/resolve/{REVISION}/{f.rfilename}?download=true",
            f"  out={f.rfilename}",
            f"  checksum=sha-256={f.lfs.sha256}",
        ]
    )
input_file = ROOT / "runs/logs/qwen3-download-input.txt"
input_file.write_text("\n".join(manifest) + "\n")
env = dict(os.environ)
env.pop("ALL_PROXY", None)
env.pop("all_proxy", None)
subprocess.run(
    [
        "aria2c",
        "--input-file=" + str(input_file),
        "--dir=" + str(MODEL),
        "--max-concurrent-downloads=3",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--continue=true",
        "--auto-file-renaming=false",
        "--check-integrity=true",
        "--max-tries=20",
        "--retry-wait=5",
        "--connect-timeout=20",
        "--timeout=60",
        "--summary-interval=30",
        "--console-log-level=warn",
    ],
    check=True,
    env=env,
)
checks = {}
for f in files:
    path = MODEL / f.rfilename
    assert path.stat().st_size == f.size
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    assert sha == f.lfs.sha256, f.rfilename
    checks[f.rfilename] = dict(size=f.size, sha256=sha)
audit = dict(
    repo_id=REPO,
    revision=REVISION,
    local_dir=str(MODEL),
    files=checks,
    downloaded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
)
(MODEL / "download-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("COMPLETE: all model shards verified against official SHA256 checksums", flush=True)
