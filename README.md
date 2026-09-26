# UNORL

Research code for low-resource, on-policy reinforcement learning and
reward-conditioned online SFT using SkyRL's single-turn math rollout and shared
math-answer verifier.

```
Sampling system: Generate a correct solution.
Training system: Generate a correct solution.     if reward > 0
                 Generate an incorrect solution. otherwise
```

Conditioning uses ordinary text in the system role, with no added control token.
Loss applies only to response tokens. Both labels receive positive cross-entropy
updates; rewards select the system instruction, with no signed RL gradients or KL.
Historical special-token runs and their source snapshots remain available.
The shared verifier returns +1/-1. It recognizes the last explicit `Answer: ...`
or `\boxed{...}` in the full response, using SkyRL answer normalization. Both
methods receive the same system instruction: end with `Answer: \boxed{...}`. Truncated responses are loss-masked using
SkyRL's overlong filtering; all-correct/all-incorrect groups are retained.

## Assets and runtime

- Base model: [Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
- Training: [DAPO-Math-8k-Stratified](https://huggingface.co/datasets/eshwarprasadS/DAPO-Math-8k-Stratified), published 7,500-row train split; the 500-row validation split is stored separately and excluded from training
- Evaluation: [AIME-2024](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024), [AIME-2025](https://huggingface.co/datasets/math-ai/aime25), [AIME-2026](https://huggingface.co/datasets/math-ai/aime26), 30 questions each
- Runtime: `../SciBuddy/.venv-skyrl/bin/python`, using its SkyRL checkout at
  `0b286bacba2bb51dfe50186b6b5d6b1e0b5f5518` (verified on each launch).
  See [SkyRL extension guidance](https://docs.skyrl.ai/docs/getting-started/development).

Assets are downloaded into this directory: `models/`, `data/raw/`, and prepared
`data/train.parquet` / `data/eval.parquet`. The original full DAPO download is
also retained under `data/raw/dapo`, but is not used in this comparison. `assets.json` pins source revisions.
The original model is preserved. The active `Qwen2.5-Math-1.5B-system`
view links the original weights and tokenizer, with context extended to 16,384
using the same RoPE settings. It does not use the historical modified embeddings.
The runtime is reused read-only from SciBuddy; this project depends on that
environment remaining available. No SciBuddy code or environment is modified.

## Run

```bash
../SciBuddy/.venv-skyrl/bin/python scripts/setup.py
python3 scripts/check.py
python3 scripts/train.py --launch --pair --smoke
python3 scripts/train.py --launch --pair
```

`--dry-run` prints the resolved configuration. `--launch` uses nohup and records
the PID, log, and eventual exit status under `runs/logs/`. Each run saves its
configuration, asset identities, source snapshot, conditioning counts, evaluation
responses, and checkpoints under `runs/<ID>/`.

The pilot configuration uses all **8 A100s**, 100 online steps, 32 prompts per
step, 8 responses per prompt, AdamW LR 1e-6, and 5 warmup steps. Each step makes
one optimizer update. Prompts are limited to 1023 tokens before the extra control
token; responses to 8192. Both methods use a configured context of 16384 tokens
(originally 4096), with the original RoPE settings and no added scaling. AIME
evaluation runs before training and every 20 steps with 8 samples per question,
720 responses per evaluation across the three years.
Two resumable checkpoints are retained; the final HF model is exported.

## GRPO comparison

`--pair` runs conditional online SFT and then standard GRPO, each on all eight
GPUs. `--mode grpo --launch` runs only the baseline. Both start from the same
prepared weights and tokenizer, with the same data order seed, sampling settings,
training budget and evaluation cadence. GRPO does not prepend a control token;
it uses group-standardized advantages and the regular PPO loss with symmetric
0.2 clipping. KL is disabled for both methods. The dummy positive advantages in
the conditional run are ignored by cross-entropy.

Each run writes `metrics.jsonl` and `evaluation.jsonl` for comparisons of reward,
AIME **avg@8**, response length, and elapsed time at matched online steps.
For each year, avg@8 = correct responses / (30 questions × 8 samples), i.e. the
mean of eight-sample correctness per question. It is reported separately for
2024/2025/2026 and overall across 90 questions. Values in JSON are fractions;
the comparison CSV also includes percentages. The underlying SkyRL pass@8 is
retained in raw metrics, but is not the requested primary report.
Each AIME year has only 30 questions, so treat small differences cautiously.

The paired launcher writes `runs/<pair-ID>/comparison.csv` after both runs finish.

Dataset audit: the published 7,500-row training split contains 7,311 unique
whitespace-normalized prompts. Its 500-row validation split has 18 prompt
overlaps with training and is not used for evaluation. The 90 AIME questions
have no exact whitespace-normalized prompt overlap with training. These checks
do not detect semantic paraphrases; counts are saved in `data/audit.json`.

The first full run was stopped because the original Answer-only grader missed
boxed answers. Its logs and frozen source remain preserved, but its reward labels
and scores are not valid comparison results. Corrected experiments start from the
prepared base weights, rather than resuming the stopped run.

`--smoke` performs one update per method with the full configured prompt batch,
rollout count, and response cap. It evaluates all three AIME years before and
after the update, disables warmup so the update has a nonzero learning rate,
and saves the training batch for inspection. It does not launch a full run afterward.

## Sampled system-prompt benchmark test

Run `python scripts/evaluate.py` to launch an evaluation-only comparison of the
base checkpoint with and without the correctness system instruction, on all 8 GPUs.
MATH-500: 500 questions, one sampled response each, accuracy.
AMC23: 40 questions, eight samples each, pass@8.
AIME25: 30 questions, eight samples each, pass@8.
All use temperature 1, top-p 1, and at most 8,192 response tokens.
Pass@8 is the fraction of questions with at least one correct response among eight.
This compares initial prompting behavior; it is not a trained GRPO comparison.

The benchmark grader selects the last explicit answer and uses
[Math-Verify](https://github.com/huggingface/Math-Verify) for mathematical equivalence.
Local dependencies are pinned in `requirements-eval.txt` and installed under `.deps`
(`python -m pip install --target .deps -r requirements-eval.txt`).
Outputs, per-benchmark metrics, configuration, and source snapshots are saved under
`runs/system-benchmarks-*/`; `results.json` updates after each benchmark.
Published MATH-500 and AMC23 benchmarks contain 2 and 6 exact question overlaps
with the current training set. Remove those training rows before future training
comparisons; this evaluation does not train on any data.

## Active Qwen3 comparison (2026-09-15)

Run pair: `qwen3-4b-instruct-pair-20260915-02`.
Model: local `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`; all weight SHA256 checks passed.
No-KL GRPO runs first, then system-conditioned online SFT. Each uses all eight
A100s for 100 updates, from the same original instruct weights; resume is disabled.
Batch: 32 prompts x 8 responses; response cap: 16,384 for training and evaluation;
prompt cap: 2,048; engine context cap: 18,432. Learning rate: 1e-6 with five warmup
steps. Both correct and incorrect conditional examples use positive cross-entropy
under their reward-selected system instructions; sampling requests correctness.

Training uses `data/train-benchmark-clean.parquet` (7,492 rows).
Evaluate before each method trains and every 20 updates: AIME25 and AIME26 pass@8 (30 problems each); temperature 1, top-p 1.
Save checkpoints every 20 updates and final HF weights at update 100.
Profile: `configs/qwen3-4b-instruct.json`. Per-run source/config/asset snapshots and
metrics live under `runs/`. Prior Qwen2.5 result artifacts were deleted at user request.

## Low-resource signed REINFORCE + rank-one ReLoRA

The new `reinforce` mode uses SkyRL rollouts, math grading, FSDP2, and inference
weight transport. It does not modify the shared SkyRL checkout.

- One fresh response per prompt; correct reward `+1`, incorrect reward `-1`.
- Loss: `-mean_over_responses(reward * sum_response_token_logprobs)`.
  No critic, reference model, KL penalty, reward centering, PPO clipping,
  positive-only filtering, or response-length normalization.
- Exactly one SGD update per rollout batch, using microbatch accumulation.
  SGD has zero momentum and weight decay; only LoRA factors receive gradients.
- Rank-one LoRA on all supported linear projections (attention and MLP,
  excluding the language-model output head), alpha 1, dropout 0.
- Every 10 successful optimizer updates: accumulate `W <- W + alpha * B @ A`,
  reinitialize A, and zero B. Parameter objects remain unchanged. The scheduler
  continues across merges; this implements periodic merging, not the original
  ReLoRA paper's entire training recipe.
- Every rollout/evaluation sync streams dense `W + alpha * B @ A` weights to
  vLLM. It includes earlier accumulated merges and the current adapter. The
  training base is not mutated by this export. The inference engine has no LoRA.

The profile is `configs/qwen3-4b-base-reinforce-relora.json`: Qwen3-4B-Base,
8 GPUs, 32 prompts/step, one rollout/prompt, 8192 response tokens, 300 steps,
AIME25/26 evaluation with eight samples every 20 steps. SGD LR `0.001` is an
initial, untuned setting; the profile uses five warmup updates and gradient norm
clipping at 1.0. Existing conditional/positive/GRPO modes are unchanged.

```bash
python scripts/train.py --mode reinforce \
  --config qwen3-4b-base-reinforce-relora.json --dry-run
# After validation, launch explicitly:
python scripts/train.py --mode reinforce \
  --config qwen3-4b-base-reinforce-relora.json --launch \
  --run-id qwen3-4b-base-reinforce-relora-01
```

Configuration is under `trainer.low_resource`: `lora_rank`, `lora_alpha`,
`merge_interval`, and `base_dtype`. The native `trainer.policy.model.lora.rank`
remains **0 for dense inference transport**; the custom worker installs the
actual training adapters using `trainer.low_resource.lora_rank=1`. Validation
rejects incompatible settings, including multiple inner updates, nonunit
sampling temperature, group filtering, and adapter-only transport.

The default frozen base storage is FP32, to preserve small accumulated updates.
Forward computation uses SkyRL mixed precision and adapter storage stays FP32.
`base_dtype=bfloat16` reduces base storage, but can round away small merges;
`relora/rounding_relative_l2` measures this error relative to the merged update.
Merges and exports materialize one matrix at a time on each worker, not a dense
copy of the entire model. Dense synchronization still costs communication;
this implementation does not claim adapter-only transfer efficiency.

Checkpoints contain the **modified base and current adapters**, scheduler, and
SkyRL trainer/RNG state. No optimizer tensors are saved (SkyRL writes an empty
optimizer dictionary). The scheduler's update count determines merge timing on
resume. HF exports are complete merged models, not standalone PEFT adapters.
Keep the same merge settings when resuming. Metrics retain the existing reward,
entropy, gradient norm, and length diagnostics, plus merge size, rounding error,
completed cycles, and optimizer-state count.

Development checks (use the existing SkyRL virtual environment):

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=.deps:.:../SciBuddy/third_party/SkyRL \
  ../SciBuddy/.venv-skyrl/bin/python -m pytest tests/test_low_resource.py -q
ruff check conditional_rl/low_resource*.py tests/test_low_resource.py scripts/train.py
ruff format --check conditional_rl/low_resource*.py tests/test_low_resource.py scripts/train.py
```
