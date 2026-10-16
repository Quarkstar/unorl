---
title: "GRPO · rank 1 alpha 1 lr 5e-4, overlong filtering off"
---

# GRPO · rank 1 alpha 1 lr 5e-4, overlong filtering off

Matched-effective-LR GRPO arm identical to the filtering-on version except generator.apply_overlong_filtering is false, so truncated responses keep their loss mask and train on their negative reward. Tests whether penalising truncation in the update stops the response-length / truncation drift seen with filtering on.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-lora-r1-a1-lr5e-4-nofilter-20261015-01` |
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

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-lora-r1-a1-lr5e-4-nofilter-20261015-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-lora-r1-a1-lr5e-4-nofilter-20261015-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.7% | 6.7% |
| 20 | 15.0% | 30.0% |
| 40 | 15.4% | 36.7% |
| 60 | 16.2% | 40.0% |
| 80 | 17.9% | 36.7% |
| 100 | 17.5% | 30.0% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 4.2% | 13.3% |
| 20 | 14.2% | 33.3% |
| 40 | 13.8% | 30.0% |
| 60 | 13.8% | 26.7% |
| 80 | 10.8% | 23.3% |
| 100 | 15.0% | 26.7% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 4/240 | 2/30 |
| aime25 | 20 | 36/240 | 9/30 |
| aime25 | 40 | 37/240 | 11/30 |
| aime25 | 60 | 39/240 | 12/30 |
| aime25 | 80 | 43/240 | 11/30 |
| aime25 | 100 | 42/240 | 9/30 |
| aime26 | 0 | 10/240 | 4/30 |
| aime26 | 20 | 34/240 | 10/30 |
| aime26 | 40 | 33/240 | 9/30 |
| aime26 | 60 | 33/240 | 8/30 |
| aime26 | 80 | 26/240 | 7/30 |
| aime26 | 100 | 36/240 | 8/30 |

## Quantitative observations

Training response correctness averaged **14.9%** over the first 10 logged updates and **37.8%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 1.011 at step 1 → 0.1269 at step 100.
- Policy gradient norm: 0.0006686 at step 1 → 0.002438 at step 100.
- Mean generated response tokens: 1193 at step 1 → 3786 at step 100.

Best recorded AIME25 pass@8: **40.0% at step 60**. Last recorded: **30.0% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
