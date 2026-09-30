---
title: "Archived · Instruct GRPO pilot"
---

# Archived · Instruct GRPO pilot

This short pilot uses an instruction-tuned model and a 16k response budget. Its initial accuracy is much higher than Base. Keep it outside Base-model algorithm comparisons; no post-training AIME evaluation is retained in this run.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-instruct-grpo-20260915-01` |
| Record | 12 steps logged / 100 planned |
| Group | Earlier research |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Instruct-2507 |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | grpo |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 0 |
| LoRA alpha | — |
| LoRA initialization | kaiming (default) |
| Learning rate | 1e-06 |
| Warmup steps | 5 |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 16384 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | not retained |

[Download the metric/configuration snapshot](../data/qwen3-4b-instruct-grpo-20260915-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-instruct-grpo-20260915-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 45.0% | 66.7% |

### AMC23

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 86.2% | 97.5% |

### MATH500

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 85.4% | — |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 108/240 | 20/30 |
| amc23 | 0 | 276/320 | 39/40 |
| math500 | 0 | 427/500 | 427/500 |

## Quantitative observations

Training response correctness averaged **69.5%** over the first 10 logged updates and **69.7%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.2322 at step 1 → 0.2513 at step 12.
- Policy gradient norm: 0.1273 at step 1 → 0.2056 at step 12.
- Mean generated response tokens: 4014 at step 1 → 4425 at step 12.

Best recorded AIME25 pass@8: **66.7% at step 0**. Last recorded: **66.7% at step 0**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
