---
title: "Single-GPU ReLoRA exploration"
---

# Single-GPU ReLoRA exploration

This combines rank-1 LoRA, periodic merge/reset, single-rollout REINFORCE, SGD and a batch of 16. It is an exploratory memory-oriented run with multiple simultaneous changes. It cannot identify which change caused its weak learning trend.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-reinforce-relora-gpu0-b16-m100-300-20260922-01` |
| Record | 295 steps logged / 300 planned |
| Group | Optimizer and loss confounds |
| Optimizer | SGD |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | signed_reinforce |
| Policy loss | signed_reinforce |
| Loss reduction | seq_mean_token_sum_norm |
| LoRA rank | 1 |
| LoRA alpha | 1 |
| Learning rate | 0.0003 |
| Warmup steps | 0 |
| Prompts × responses | 16 × 1 = 16 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 4 |
| GPUs (policy) | 1 |
| KL loss / reward | False / False |
| Exit status | 1 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-reinforce-relora-gpu0-b16-m100-300-20260922-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-reinforce-relora-gpu0-b16-m100-300-20260922-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 4.2% | 20.0% |
| 100 | 3.3% | 20.0% |
| 200 | 1.7% | 6.7% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 10/240 | 6/30 |
| aime25 | 100 | 8/240 | 6/30 |
| aime25 | 200 | 4/240 | 2/30 |

## Quantitative observations

Training response correctness averaged **8.1%** over the first 10 logged updates and **7.5%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 1.245 at step 1 → 0.9627 at step 295.
- Policy gradient norm: 10.35 at step 1 → 12.18 at step 295.
- Mean generated response tokens: 1403 at step 1 → 1686 at step 295.

Best recorded AIME25 pass@8: **20.0% at step 0**. Last recorded: **6.7% at step 200**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
