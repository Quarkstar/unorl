---
title: "GRPO · full-layer rank-1 LoFT-simple"
---

# GRPO · full-layer rank-1 LoFT-simple

Launched a 100-step GRPO comparison using the authors' LoFT-simple optimizer geometry: alternate B/A updates starting with B; rescale gradients and transport first moments; update moments for both factors every step; no second-moment transport or merge. Retain the reference's Kaiming initialization, LR 1.5e-5, beta values, batch, eight rollouts and response budget. Method-required differences include alpha 1 and Adam epsilon 1e-4, versus alpha 32 / epsilon 1e-8 in the reference. Rank-one specialization avoids dense weight-shaped calibration tensors; saved previous factors and alternation phase are part of the checkpoint. Eight-step CPU parity with the pinned authors' implementation and clipping agreed to within 1.5e-8 in parameters. Actual eight-GPU FSDP2 checks verified frozen backbone, A/B optimizer state, adapter export and exact next-update reproduction after resume. Calibrated active gradient norms are not directly comparable to ordinary LoRA norms. Completed 100 steps successfully. Successive 20-step training-correctness means were 8.20%, 9.24%, 8.46%, 9.22%, and 8.89%; this run did not establish meaningful learning. Final AIME25 avg@8 / pass@8 was 3.33% / 20.0%, versus 20.0% / 43.3% for the standard LoRA reference. Peak training allocated / reserved memory was 17.75 / 18.60 GiB. Checkpoints and export weights were removed after completion; measurements retained. Results do not establish that the authors' method generally fails.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-loft-simple-r1-20261001-01` |
| Record | 100 steps logged / 100 planned |
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
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

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
| 20 | 2.1% | 10.0% |
| 40 | 0.4% | 3.3% |
| 60 | 1.2% | 10.0% |
| 80 | 2.1% | 13.3% |
| 100 | 3.3% | 20.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 5/240 | 4/30 |
| aime25 | 20 | 5/240 | 3/30 |
| aime25 | 40 | 1/240 | 1/30 |
| aime25 | 60 | 3/240 | 3/30 |
| aime25 | 80 | 5/240 | 4/30 |
| aime25 | 100 | 8/240 | 6/30 |

## Quantitative observations

Training response correctness averaged **7.7%** over the first 10 logged updates and **7.7%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.7889 at step 1 → 0.7868 at step 100.
- Policy gradient norm: 0.002852 at step 1 → 6.881 at step 100.
- Mean generated response tokens: 1026 at step 1 → 1221 at step 100.

Best recorded AIME25 pass@8: **20.0% at step 100**. Last recorded: **20.0% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
