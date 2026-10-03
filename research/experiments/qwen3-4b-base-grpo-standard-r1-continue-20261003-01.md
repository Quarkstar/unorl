---
title: "GRPO · standard rank-1 LoRA continuation (shared step-100 checkpoint)"
---

# GRPO · standard rank-1 LoRA continuation (shared step-100 checkpoint)

Matched control for the gradual-refresh candidate. Load the historical standard rank-one LoRA global-step-100 model, native AdamW moments/counters, training RNG shards and dataloader position. Train 100 additional updates, global steps 101–200. Disable all merges and refreshes; use the same worker, native optimizer, scheduler, rollout synchronization, monitoring and diagnostic plumbing as the candidate. Rank one / alpha 32, LR 1.5e-5, eight rollouts, 32 prompts/update, 8192 response tokens, no KL, no warmup. AIME25 sampled avg@8/pass@8 every 20 steps. Save resumable checkpoints every 20 steps, retaining the newest; preserve the shared source checkpoint. Initial evaluation is at global step 100. Fresh inference engines are initialized with matching configuration, but their historical sampling RNG is not restored, so sampling is not bitwise continuation. Hourly process/metric checks and ten-second NVML samples are enabled. Curves and tables below reflect the latest retained metrics; outcome analysis follows completion. This is an advanced-policy continuation, not a new 0–100 trial.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-standard-r1-continue-20261003-01` |
| Record | 100 new updates; global step 200 / 200 |
| Group | Matched continuation from the standard-LoRA step-100 checkpoint |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | grpo |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 32 |
| LoRA initialization | kaiming (default) |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Restart warmup (updates) | 0 |
| Merge/reset interval (updates) | 40 |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |
| Merges enabled | False |
| First merge global step | 101 |
| Refresh angle (degrees) | 0.0 |
| Resume checkpoint | runs/qwen3-4b-base-grpo-lora-r1-blog-20260923-01/checkpoints/global_step_100 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-standard-r1-continue-20261003-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-standard-r1-continue-20261003-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 100 | 16.7% | 33.3% |
| 120 | 19.2% | 40.0% |
| 140 | 17.9% | 40.0% |
| 160 | 19.6% | 40.0% |
| 180 | 15.0% | 26.7% |
| 200 | 17.1% | 33.3% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 100 | 40/240 | 10/30 |
| aime25 | 120 | 46/240 | 12/30 |
| aime25 | 140 | 43/240 | 12/30 |
| aime25 | 160 | 47/240 | 12/30 |
| aime25 | 180 | 36/240 | 8/30 |
| aime25 | 200 | 41/240 | 10/30 |

## Quantitative observations

Training response correctness averaged **36.2%** over the first 10 logged updates and **39.0%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.1225 at step 101 → 0.1036 at step 200.
- Policy gradient norm: 0.04803 at step 101 → 0.04158 at step 200.
- Mean generated response tokens: 3865 at step 101 → 4645 at step 200.

Best recorded AIME25 pass@8: **40.0% at step 120**. Last recorded: **33.3% at step 200**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
