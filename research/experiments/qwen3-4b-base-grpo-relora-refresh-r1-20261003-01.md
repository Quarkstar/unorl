---
title: "GRPO · compensated gradual refresh, rank 1 (100 steps from base)"
---

# GRPO · compensated gradual refresh, rank 1 (100 steps from base)

Main-budget trial of the proposed continuity-preserving refresh algorithm. Start from Qwen3-4B-Base, not a trained checkpoint; train exactly 100 updates with the standard rank-one LoRA GRPO recipe. Refresh after updates 40 and 80: rotate A by 20 degrees, retain B, compensate the frozen backbone, retain Adam counters and A moments, project B first moments by cosine and retain B variances. Moment transport is approximate, not a convergence guarantee. Native AdamW LR 1.5e-5 constant, alpha 32, eight rollouts, 32 prompts per update, 8192 response tokens, all eight GPUs. AIME25 avg@8 and pass@8 every twenty updates; hourly monitoring and ten-second VRAM sampling. Compare the complete 0-100 reward/evaluation curve with the historical standard rank-one reference and cold-reset trials, particularly steps 46-60. No result or superiority is assumed before completion.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-relora-refresh-r1-20261003-01` |
| Record | 91 steps logged / 100 planned |
| Group | Controlled follow-up experiments |
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
| First merge global step | 40 |
| Refresh angle (degrees) | 20.0 |
| Resume checkpoint | None |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-relora-refresh-r1-20261003-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-relora-refresh-r1-20261003-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 3.3% | 13.3% |
| 20 | 5.8% | 20.0% |
| 40 | 17.1% | 26.7% |
| 60 | 17.5% | 36.7% |
| 80 | 17.1% | 30.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 8/240 | 4/30 |
| aime25 | 20 | 14/240 | 6/30 |
| aime25 | 40 | 41/240 | 8/30 |
| aime25 | 60 | 42/240 | 11/30 |
| aime25 | 80 | 41/240 | 9/30 |

## Quantitative observations

Training response correctness averaged **9.7%** over the first 10 logged updates and **41.3%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8753 at step 1 → 0.1373 at step 91.
- Policy gradient norm: 0.02517 at step 1 → 0.03368 at step 91.
- Mean generated response tokens: 1339 at step 1 → 3782 at step 91.

Best recorded AIME25 pass@8: **36.7% at step 60**. Last recorded: **30.0% at step 80**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
