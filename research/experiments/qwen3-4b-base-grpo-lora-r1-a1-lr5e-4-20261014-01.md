---
title: "GRPO · rank 1 alpha 1 lr 5e-4"
---

# GRPO · rank 1 alpha 1 lr 5e-4

Matched-effective-LR arm: GRPO, rank-1 LoRA alpha 1 (scale 1), LR 5e-4, so alpha/rank x lr is about the alpha-32 baseline's 32 x 1.5e-5 = 4.8e-4. 32 prompts x 8 rollouts, token_mean, no KL, no clipping, 100 updates, AIME25+AIME26 avg@8 every 20, logs only. Learns much faster than the alpha-32 baseline at matched nominal effective LR, indicating the alpha/LR split matters under AdamW beyond their product.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-lora-r1-a1-lr5e-4-20261014-01` |
| Record | 100 steps logged / 100 planned |
| Group | Controlled follow-up experiments |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | grpo |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 1 |
| LoRA initialization | kaiming (default) |
| Learning rate | 0.0005 |
| Warmup steps | 0 |
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-lora-r1-a1-lr5e-4-20261014-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-lora-r1-a1-lr5e-4-20261014-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.9% | 13.3% |
| 20 | 14.2% | 26.7% |
| 40 | 16.7% | 26.7% |
| 60 | 19.6% | 36.7% |
| 80 | 20.4% | 33.3% |
| 100 | 20.4% | 40.0% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.5% | 16.7% |
| 20 | 15.0% | 33.3% |
| 40 | 17.5% | 30.0% |
| 60 | 14.6% | 26.7% |
| 80 | 16.2% | 30.0% |
| 100 | 15.4% | 30.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 7/240 | 4/30 |
| aime25 | 20 | 34/240 | 8/30 |
| aime25 | 40 | 40/240 | 8/30 |
| aime25 | 60 | 47/240 | 11/30 |
| aime25 | 80 | 49/240 | 10/30 |
| aime25 | 100 | 49/240 | 12/30 |
| aime26 | 0 | 6/240 | 5/30 |
| aime26 | 20 | 36/240 | 10/30 |
| aime26 | 40 | 42/240 | 9/30 |
| aime26 | 60 | 35/240 | 8/30 |
| aime26 | 80 | 39/240 | 9/30 |
| aime26 | 100 | 37/240 | 9/30 |

## Quantitative observations

Training response correctness averaged **15.7%** over the first 10 logged updates and **40.5%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8647 at step 1 → 0.1268 at step 100.
- Policy gradient norm: 0.001276 at step 1 → 0.005336 at step 100.
- Mean generated response tokens: 1309 at step 1 → 4703 at step 100.

Best recorded AIME25 pass@8: **40.0% at step 100**. Last recorded: **40.0% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
