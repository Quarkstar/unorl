---
title: "GRPO · gradual A refresh with warm B (shared step-100 checkpoint)"
---

# GRPO · gradual A refresh with warm B (shared step-100 checkpoint)

Candidate derived from the adapter-gradient discontinuity analysis. Start from exactly the same historical standard-LoRA step-100 checkpoint as the control, restoring native AdamW history and dataloader position. Train global steps 101–200; after optimizer updates 101, 141 and 181, rotate each rank-one A row by 20 degrees toward a fresh orthogonal direction at the same row norm. Keep B unchanged and compensate W += scale * B @ (A_old - A_new), preserving the effective weight in exact arithmetic. Retain A moments and all Adam counters, project B first moments by cos(20 degrees), and retain B variances. The variance and first-moment treatment is approximate; unobserved orthogonal-gradient history is not reconstructed. Constant LR, no reset warmup. All remaining recipe and monitoring settings match the control. Log real response-prefix KL, base corrections, accumulated rank, allocator peaks and hourly health; factor history is checkpointed with model/Adam state. Initial evaluation at global step 100 precedes the first intervention. Curves/tables are updated from retained metrics; outcome analysis follows completion. No guarantee of improvement or exact mixed-precision continuity is claimed.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-relora-refresh-r1-continue-20261003-01` |
| Record | 13 new updates; global step 113 / 200 |
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
| Exit status | not retained |
| Merges enabled | True |
| First merge global step | 101 |
| Refresh angle (degrees) | 20.0 |
| Resume checkpoint | runs/qwen3-4b-base-grpo-lora-r1-blog-20260923-01/checkpoints/global_step_100 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-relora-refresh-r1-continue-20261003-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-relora-refresh-r1-continue-20261003-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 100 | 18.8% | 43.3% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 100 | 45/240 | 13/30 |

## Quantitative observations

Training response correctness averaged **36.0%** over the first 10 logged updates and **34.0%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.1261 at step 101 → 0.1277 at step 113.
- Policy gradient norm: 0.03747 at step 101 → 0.03779 at step 113.
- Mean generated response tokens: 3476 at step 101 → 3248 at step 113.

Best recorded AIME25 pass@8: **43.3% at step 100**. Last recorded: **43.3% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
