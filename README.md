# UNORL

Resource-efficient **on-policy reinforcement learning** with SkyRL. We study
single-rollout learning, rank-1 LoRA, and ways to reduce training activation memory.

The [published research book](https://quarkstar.github.io/unorl/) contains a page for every retained experiment,
configuration snapshots, Material-colored learning curves, and a comparison of results.
Its source is in [research/](research/index.md).

| Qwen3-4B-Base, rank-1 LoRA, step 100 | AIME25 avg@8 | AIME25 pass@8 |
|---|---:|---:|
| GRPO | 20.0% | 43.3% |
| Vanilla REINFORCE / AdamW | 14.6% | 26.7% |
| Batch-normalized REINFORCE / AdamW | 17.9% | 30.0% |

These are single-run results on 30 AIME25 questions. GRPO uses 32 prompts × 8
responses, while REINFORCE uses 256 prompts × 1 response per update. Matching
response count does not match prompt exposure or total token compute. Read the
[measurement conventions](research/methods.md) before comparing experiments.

## Repository layout

```text
unorl/                 SkyRL trainers, rewards, prompts, LoRA and PPO extensions
configs/               Portable training profiles
scripts/train.py       Audited training launcher
scripts/research/      Snapshot exporter and book/plot generator
research/              MyST book, individual experiments, data and figures
research/notes/        Paper and method notes
tests/                 CPU checks for algorithms and documentation integrity
.github/workflows/     Lint, documentation checks and GitHub Pages deployment
```

`models/`, `data/`, `runs/`, caches and rendered HTML are local and ignored by Git.
Published research snapshots contain metrics and configurations, not model weights
or generated solutions. Historical records retain their original run identifiers.
The active Python package and runtime variables are named `unorl` / `UNORL_*`.

## Read or build the book

Requires Python 3.12+ and Node.js 22+ (tested with Node 24).

```bash
python -m venv .venv-docs
.venv-docs/bin/pip install -r requirements-docs.txt
npm ci
.venv-docs/bin/python scripts/research/build.py
npm run docs:build
```

The static book is written to `_build/html/`. For local browsing:

```bash
python -m http.server 8000 --directory _build/html
# Or run the MyST development server:
npm run docs:dev
```

The committed snapshots make the book buildable on a fresh clone without GPUs.
See [publishing](research/publishing.md) for GitHub Pages setup.

## Training

Training uses a **separately installed SkyRL runtime**, pinned to commit
`0b286bacba2bb51dfe50186b6b5d6b1e0b5f5518`. Installing `unorl` alone does not
install CUDA, vLLM, SkyRL or models. The launcher checks the SkyRL revision before
training. On the original machine it can reuse `../SciBuddy`; elsewhere set:

```bash
export SKYRL_ROOT=/path/to/SkyRL
export SKYRL_PYTHON=/path/to/skyrl-environment/bin/python
```

Place Qwen3-4B-Base under `models/Qwen3-4B-Base/` and prepare the benchmark-clean
math data at `data/train-benchmark-clean.parquet` and `data/eval-aime25.parquet`.
The training dataset is derived from `eshwarprasadS/DAPO-Math-8k-Stratified`;
source revisions and data audits for the recorded runs are preserved locally.
The schema and runtime details are described in [methods](research/methods.md).
The download helper is `scripts/download_qwen3.py` (use `--repo Qwen/Qwen3-4B-Base
--output models/Qwen3-4B-Base`). It does not prepare training data.

```bash
# Inspect the configuration without launching training or requiring a GPU:
python scripts/train.py --mode reinforce \
  --config qwen3-4b-base-reinforce-batchnorm-adamw-r1.json --dry-run

# Explicit GPU selection; this profile expects eight policy GPUs:
python scripts/train.py --mode reinforce \
  --config qwen3-4b-base-reinforce-batchnorm-adamw-r1.json \
  --gpus 0,1,2,3,4,5,6,7 --launch
```

Configs use relative asset paths, resolved against `UNORL_PROJECT_ROOT` (the
repository root by default). The launcher saves resolved config, source snapshot,
SkyRL/project revisions, metrics and logs for each new run. Existing run IDs are
never silently reused. The original CUDA runtime settings are retained; configure
the runtime for your hardware before launching elsewhere.

Retired conditional-SFT training and migration scripts have been removed from the
active code. Their substantive experiment results remain in the book. Smoke-run
records were removed; the [cleanup manifest](research/maintenance/cleanup.json)
records exactly which paths were deleted.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks, experiment recording, and the
reproducibility expectations. The [GitHub repository](https://github.com/Quarkstar/unorl)
builds and publishes the research book through GitHub Actions on pushes to `main`.

## License

UNORL is licensed under the [MIT License](LICENSE). Third-party libraries, models
and datasets retain their own licenses.
