---
title: "Raw signed REINFORCE · rank 1 alpha 1 lr 5e-4"
---

# Raw signed REINFORCE · rank 1 alpha 1 lr 5e-4

Matched-effective-LR estimator arm: raw signed REINFORCE on the terminal +1/-1 reward with no centering or whitening (unorl signed_reinforce estimator and loss), 256 prompts x 1, rank-1 LoRA alpha 1, LR 5e-4, token_mean.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-reinforce-adamw-r1-a1-lr5e-4-20261014-01` |
| Record | 100 steps logged / 100 planned |
| Group | Controlled follow-up experiments |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | signed_reinforce |
| Policy loss | signed_reinforce |
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

[Download the metric/configuration snapshot](../data/qwen3-4b-base-reinforce-adamw-r1-a1-lr5e-4-20261014-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-reinforce-adamw-r1-a1-lr5e-4-20261014-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.1% | 13.3% |
| 20 | 12.5% | 33.3% |
| 40 | 15.0% | 36.7% |
| 60 | 17.9% | 26.7% |
| 80 | 11.7% | 23.3% |
| 100 | 17.1% | 30.0% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 0.8% | 6.7% |
| 20 | 13.8% | 26.7% |
| 40 | 11.7% | 26.7% |
| 60 | 16.7% | 33.3% |
| 80 | 8.8% | 20.0% |
| 100 | 17.1% | 23.3% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 5/240 | 4/30 |
| aime25 | 20 | 30/240 | 10/30 |
| aime25 | 40 | 36/240 | 11/30 |
| aime25 | 60 | 43/240 | 8/30 |
| aime25 | 80 | 28/240 | 7/30 |
| aime25 | 100 | 41/240 | 9/30 |
| aime26 | 0 | 2/240 | 2/30 |
| aime26 | 20 | 33/240 | 8/30 |
| aime26 | 40 | 28/240 | 8/30 |
| aime26 | 60 | 40/240 | 10/30 |
| aime26 | 80 | 21/240 | 6/30 |
| aime26 | 100 | 41/240 | 7/30 |

## Quantitative observations

Training response correctness averaged **10.4%** over the first 10 logged updates and **36.0%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8505 at step 1 → 0.05142 at step 100.
- Policy gradient norm: 0.00278 at step 1 → 0.0173 at step 100.
- Mean generated response tokens: 1138 at step 1 → 4505 at step 100.

Best recorded AIME25 pass@8: **36.7% at step 40**. Last recorded: **30.0% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
