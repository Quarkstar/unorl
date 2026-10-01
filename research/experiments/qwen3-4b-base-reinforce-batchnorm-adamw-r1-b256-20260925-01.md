---
title: "Batch-normalized REINFORCE · AdamW"
---

# Batch-normalized REINFORCE · AdamW

Learning improves rapidly, then training correctness levels off. Final AIME25 avg@8 is slightly higher than at step 60, while pass@8 falls: correct responses cover fewer questions. Sampling variation and changes in per-question competence cannot be separated from a single evaluation draw. The small final advantage over vanilla is not a statistically established ranking.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-reinforce-batchnorm-adamw-r1-b256-20260925-01` |
| Record | 100 steps logged / 100 planned |
| Group | Current algorithm comparison |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | batch_norm_reinforce |
| Policy loss | signed_reinforce |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 32 |
| LoRA initialization | kaiming (default) |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 256 × 1 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-reinforce-batchnorm-adamw-r1-b256-20260925-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-reinforce-batchnorm-adamw-r1-b256-20260925-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.2% | 6.7% |
| 20 | 6.7% | 26.7% |
| 40 | 14.6% | 30.0% |
| 60 | 17.5% | 43.3% |
| 80 | 12.9% | 16.7% |
| 100 | 17.9% | 30.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 3/240 | 2/30 |
| aime25 | 20 | 16/240 | 8/30 |
| aime25 | 40 | 35/240 | 9/30 |
| aime25 | 60 | 42/240 | 13/30 |
| aime25 | 80 | 31/240 | 5/30 |
| aime25 | 100 | 43/240 | 9/30 |

## Quantitative observations

Training response correctness averaged **10.1%** over the first 10 logged updates and **37.1%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8407 at step 1 → 0.1223 at step 100.
- Policy gradient norm: 0.08121 at step 1 → 0.05439 at step 100.
- Mean generated response tokens: 1254 at step 1 → 4545 at step 100.

Best recorded AIME25 pass@8: **43.3% at step 60**. Last recorded: **30.0% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
