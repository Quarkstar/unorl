---
title: "GRPO · rank 1 alpha 1 lr 1e-5"
---

# GRPO · rank 1 alpha 1 lr 1e-5

Matched estimator comparison, all rank-1 LoRA with alpha 1 (scale 1), native AdamW constant LR 1e-5, no KL, no clipping, token_mean, 256 responses per update, DAPO math, 100 updates, AIME25+AIME26 avg@8 every 20. This arm: GRPO with 32 prompts x 8 rollouts and the per-prompt group-mean baseline.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-lora-r1-a1-20261013-01` |
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
| Learning rate | 1e-05 |
| Warmup steps | 0 |
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-lora-r1-a1-20261013-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-lora-r1-a1-20261013-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.1% | 13.3% |
| 20 | 2.9% | 13.3% |
| 40 | 1.7% | 6.7% |
| 60 | 2.9% | 16.7% |
| 80 | 4.2% | 20.0% |
| 100 | 2.5% | 13.3% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 0.8% | 6.7% |
| 20 | 2.5% | 16.7% |
| 40 | 1.2% | 10.0% |
| 60 | 1.2% | 6.7% |
| 80 | 3.3% | 20.0% |
| 100 | 2.5% | 13.3% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 5/240 | 4/30 |
| aime25 | 20 | 7/240 | 4/30 |
| aime25 | 40 | 4/240 | 2/30 |
| aime25 | 60 | 7/240 | 5/30 |
| aime25 | 80 | 10/240 | 6/30 |
| aime25 | 100 | 6/240 | 4/30 |
| aime26 | 0 | 2/240 | 2/30 |
| aime26 | 20 | 6/240 | 5/30 |
| aime26 | 40 | 3/240 | 3/30 |
| aime26 | 60 | 3/240 | 2/30 |
| aime26 | 80 | 8/240 | 6/30 |
| aime26 | 100 | 6/240 | 4/30 |

## Quantitative observations

Training response correctness averaged **8.9%** over the first 10 logged updates and **8.7%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.9447 at step 1 → 0.8005 at step 100.
- Policy gradient norm: 0.001976 at step 1 → 0.00151 at step 100.
- Mean generated response tokens: 1246 at step 1 → 1261 at step 100.

Best recorded AIME25 pass@8: **20.0% at step 80**. Last recorded: **13.3% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
