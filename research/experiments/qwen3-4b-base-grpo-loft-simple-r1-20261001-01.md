---
title: "GRPO · full-layer rank-1 LoFT-simple"
---

# GRPO · full-layer rank-1 LoFT-simple

Launched a 100-step GRPO comparison using the authors' LoFT-simple optimizer geometry: alternate B/A updates starting with B; rescale gradients and transport first moments; update moments for both factors every step; no second-moment transport or merge. Retain the reference's Kaiming initialization, LR 1.5e-5, beta values, batch, eight rollouts and response budget. Method-required differences include alpha 1 and Adam epsilon 1e-4, versus alpha 32 / epsilon 1e-8 in the reference. Rank-one specialization avoids dense weight-shaped calibration tensors; saved previous factors and alternation phase are part of the checkpoint. Eight-step CPU parity with the pinned authors' implementation and clipping agreed to within 1.5e-8 in parameters. Actual eight-GPU FSDP2 checks verified frozen backbone, A/B optimizer state, adapter export and exact next-update reproduction after resume. Calibrated active gradient norms are not directly comparable to ordinary LoRA norms. Learning and peak-memory conclusions remain pending.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-loft-simple-r1-20261001-01` |
| Record | 0 steps logged / 100 planned |
| Group | Controlled follow-up experiments |
| Optimizer | LoFTSimpleAdamW (Adam-family) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | grpo |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 1 |
| LoRA initialization | loft_simple |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | not retained |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-loft-simple-r1-20261001-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-loft-simple-r1-20261001-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.1% | 13.3% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 5/240 | 4/30 |

## Quantitative observations


Best recorded AIME25 pass@8: **13.3% at step 0**. Last recorded: **13.3% at step 0**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
