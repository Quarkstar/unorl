---
title: "REINFORCE++ · rank 1 alpha 1 lr 5e-4, overlong filtering off"
---

# REINFORCE++ · rank 1 alpha 1 lr 5e-4, overlong filtering off

Single-rollout REINFORCE++, rank-1 LoRA alpha 1, LR 5e-4, token_mean, identical to the filtering-on arm except apply_overlong_filtering is false.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-reinforce-plus-plus-r1-a1-lr5e-4-nofilter-20261015-01` |
| Record | 100 steps logged / 100 planned |
| Group | Controlled follow-up experiments |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | reinforce++ |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 1 |
| LoRA initialization | kaiming (default) |
| Learning rate | 0.0005 |
| Warmup steps | 0 |
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 256 × 1 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-reinforce-plus-plus-r1-a1-lr5e-4-nofilter-20261015-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-reinforce-plus-plus-r1-a1-lr5e-4-nofilter-20261015-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.5% | 13.3% |
| 20 | 12.5% | 26.7% |
| 40 | 19.2% | 33.3% |
| 60 | 17.5% | 33.3% |
| 80 | 5.4% | 13.3% |
| 100 | 16.2% | 33.3% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.1% | 10.0% |
| 20 | 17.9% | 33.3% |
| 40 | 14.6% | 30.0% |
| 60 | 15.4% | 26.7% |
| 80 | 7.1% | 16.7% |
| 100 | 15.0% | 20.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 6/240 | 4/30 |
| aime25 | 20 | 30/240 | 8/30 |
| aime25 | 40 | 46/240 | 10/30 |
| aime25 | 60 | 42/240 | 10/30 |
| aime25 | 80 | 13/240 | 4/30 |
| aime25 | 100 | 39/240 | 10/30 |
| aime26 | 0 | 5/240 | 3/30 |
| aime26 | 20 | 43/240 | 10/30 |
| aime26 | 40 | 35/240 | 9/30 |
| aime26 | 60 | 37/240 | 8/30 |
| aime26 | 80 | 17/240 | 5/30 |
| aime26 | 100 | 36/240 | 6/30 |

## Quantitative observations

Training response correctness averaged **13.6%** over the first 10 logged updates and **39.9%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.9836 at step 1 → 0.09727 at step 100.
- Policy gradient norm: 0.002044 at step 1 → 0.008068 at step 100.
- Mean generated response tokens: 1141 at step 1 → 5031 at step 100.

Best recorded AIME25 pass@8: **33.3% at step 40**. Last recorded: **33.3% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
