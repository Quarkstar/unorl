---
title: "Full-parameter GRPO"
---

# Full-parameter GRPO

Full-policy GRPO provides the original learning reference: AIME25 avg@8 / pass@8 is 17.9% / 36.7% at step 100. It uses LR 1e-6 and five warmup steps; the later LoRA profiles use LR 1.5e-5 without warmup. Advantage standard-deviation normalization, policy clipping and importance correction also differ. This historical result must remain visible in comparisons, but it is not an isolated adapter ablation.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-20260916-01` |
| Record | 100 steps logged / 100 planned |
| Group | Full-parameter reference |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | grpo |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 0 |
| LoRA alpha | — |
| LoRA initialization | kaiming (default) |
| Learning rate | 1e-06 |
| Warmup steps | 5 |
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-20260916-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-20260916-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.1% | 13.3% |
| 20 | 5.4% | 23.3% |
| 40 | 12.5% | 26.7% |
| 60 | 16.2% | 26.7% |
| 80 | 16.7% | 36.7% |
| 100 | 17.9% | 36.7% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.2% | 10.0% |
| 20 | 7.5% | 30.0% |
| 40 | 13.8% | 33.3% |
| 60 | 15.0% | 26.7% |
| 80 | 13.8% | 26.7% |
| 100 | 16.7% | 30.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 5/240 | 4/30 |
| aime25 | 20 | 13/240 | 7/30 |
| aime25 | 40 | 30/240 | 8/30 |
| aime25 | 60 | 39/240 | 8/30 |
| aime25 | 80 | 40/240 | 11/30 |
| aime25 | 100 | 43/240 | 11/30 |
| aime26 | 0 | 3/240 | 3/30 |
| aime26 | 20 | 18/240 | 9/30 |
| aime26 | 40 | 33/240 | 10/30 |
| aime26 | 60 | 36/240 | 8/30 |
| aime26 | 80 | 33/240 | 8/30 |
| aime26 | 100 | 40/240 | 9/30 |

## Quantitative observations

Training response correctness averaged **8.2%** over the first 10 logged updates and **35.8%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.9277 at step 1 → 0.1039 at step 100.
- Policy gradient norm: 0.1942 at step 1 → 0.1451 at step 100.
- Mean generated response tokens: 1125 at step 1 → 4688 at step 100.

Best recorded AIME25 pass@8: **36.7% at step 80**. Last recorded: **36.7% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
