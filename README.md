# UNORL

Resource-efficient **on-policy reinforcement learning** with SkyRL. We study
single-rollout learning, rank-1 LoRA, and ways to reduce training activation memory.

The [published research book](https://quarkstar.github.io/unorl/) contains a page for every retained experiment,
configuration snapshots, Material-colored learning curves, and a comparison of results.
Its source is in [research/](research/index.md).

| Qwen3-4B-Base, step 100 | AIME25 avg@8 | AIME25 pass@8 |
|---|---:|---:|
| GRPO / full-parameter fine-tuning (historical reference) | 17.9% | 36.7% |
| GRPO / full-layer rank-1 LoRA | 20.0% | 43.3% |
| GRPO / full-layer rank-1 NoRA-init | 17.9% | 36.7% |
| GRPO / full NoRA-init with merge/reset | 15.0% | 30.0% |
| GRPO / full-layer rank-1 LoFT-simple | Running | Running |
| GRPO / full-layer rank-1 LoRA-FA | 17.5% | 33.3% |
| GRPO / LoRA in final 18 layers | 17.9% | 33.3% |
| GRPO / NoRA-init in final 18 layers | 15.8% | 30.0% |
| Vanilla REINFORCE / AdamW | 14.6% | 26.7% |
| Batch-normalized REINFORCE / AdamW | 17.9% | 30.0% |

These are single-run results on 30 AIME25 questions. GRPO uses 32 prompts × 8
responses, while REINFORCE uses 256 prompts × 1 response per update. Matching
response count does not match prompt exposure or total token compute. Read the
[measurement conventions](research/methods.md) before comparing experiments.
The historical full-parameter reference differs in learning rate, warmup,
advantage normalization, clipping and importance correction; it is not an
isolated comparison of full-parameter training against adapters.

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

## Current follow-up

Full-layer rank-one **LoFT-simple** tests gradient and first-moment calibration
with alternating A/B updates. The 100-step GRPO recipe retains eight rollouts
and the response budget; unit adapter scaling and Adam epsilon 1e-4 follow the
authors' implementation. See the [implementation and validation record](research/notes/lora-training-2025-2026.md).
The completed NoRA merge/reset trial did not sustain improvement after resets.

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

# LoFT-simple follow-up (eight GPUs):
python scripts/train.py --mode grpo \
  --config qwen3-4b-base-grpo-loft-simple-r1.json \
  --gpus 0,1,2,3,4,5,6,7 --launch

# Full NoRA-init merge/reset follow-up (eight GPUs):
python scripts/train.py --mode nora-merge \
  --config qwen3-4b-base-grpo-nora-merge-r1.json \
  --gpus 0,1,2,3,4,5,6,7 --launch

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

The full-layer rank-one LoRA-FA comparison uses
`configs/qwen3-4b-base-grpo-lorafa-r1.json`. It retains the GRPO reference's
Kaiming initialization, alpha 32, AdamW LR 1.5e-5, eight rollouts and 100 steps.
A is frozen before FSDP2; B gradients receive the regularized inverse-Gram
correction before clipping and native AdamW. The `lorafa` initialization field
selects this worker; actual initialization remains Kaiming. Run it with
`--mode grpo --config qwen3-4b-base-grpo-lorafa-r1.json --gpus 0,1,2,3,4,5,6,7`.
See the [method investigation](research/notes/lora-training-2025-2026.md).

`scripts/check_lorafa_fsdp.py` checks this strategy with a tiny Qwen3 model
under `torchrun`, including frozen weights, Adam state, checkpoint resume and
native adapter export. Its output belongs under `.tmp/`. LoRA-FA records
per-rank training allocator peaks in `training-memory-rank*.jsonl`; the
separate `scripts/monitor_vram.py` sampler measures total device residency.

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

### ReLoRA restart comparison

Use `scripts/run_relora_comparison.py --tag UNIQUE_SUFFIX` with the SkyRL
Python environment to run two matched 100-step trials on GPUs 0–7: five-update
post-merge LR ramp, then constant LR. Both use the standard Kaiming rank-one
LoRA GRPO recipe, native AdamW and merges at steps 40/80 with full Adam-state
clearing. Initial warmup remains zero. Actual math-response-prefix KL and
log-probability changes, accumulated-update spectra, and peak training memory
are recorded. The control starts only after successful completion of the ramp
trial. Successful runs remove checkpoint/export weights and temporary factor
caches, retaining logs, evaluations and diagnostic JSON records.

Validation: `scripts/check_relora_fsdp.py --output OUTPUT` under eight-process
`torch.distributed.run` checks reset scheduling, merge behavior, exact
next-update reproduction after checkpoint loading, and dense export.
