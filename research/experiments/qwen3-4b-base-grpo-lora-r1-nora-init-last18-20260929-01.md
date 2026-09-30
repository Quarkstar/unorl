---
title: "GRPO · rank-1 NoRA-init in the final 18 layers"
---

# GRPO · rank-1 NoRA-init in the final 18 layers

Completed 100 steps successfully. Relative to full-layer NoRA-init, only adapter placement changes: layers 18–35. Training correctness increases from 11.9% in steps 1–20 to 37.3% in steps 81–100. Final AIME25 avg@8 / pass@8 is 15.8% / 30.0%. The half-layer trial does not reproduce the full-layer NoRA-init run's early acceleration. Comparisons with standard initialization also change alpha from 32 to 1. This single run does not isolate the cause of the later plateau or establish a reliable ranking on 30 evaluation questions. Sampled training device-memory peak is 23.24 GiB; the ten-second NVML sampler can miss brief peaks and includes all resident processes.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-lora-r1-nora-init-last18-20260929-01` |
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
| LoRA initialization | nora_init |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-lora-r1-nora-init-last18-20260929-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-lora-r1-nora-init-last18-20260929-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.9% | 13.3% |
| 20 | 5.0% | 26.7% |
| 40 | 12.1% | 26.7% |
| 60 | 14.2% | 36.7% |
| 80 | 16.2% | 30.0% |
| 100 | 15.8% | 30.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 7/240 | 4/30 |
| aime25 | 20 | 12/240 | 8/30 |
| aime25 | 40 | 29/240 | 8/30 |
| aime25 | 60 | 34/240 | 11/30 |
| aime25 | 80 | 39/240 | 9/30 |
| aime25 | 100 | 38/240 | 9/30 |

## Quantitative observations

Training response correctness averaged **9.1%** over the first 10 logged updates and **35.6%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.6997 at step 1 → 0.1161 at step 100.
- Policy gradient norm: 0.07642 at step 1 → 0.06032 at step 100.
- Mean generated response tokens: 1166 at step 1 → 4019 at step 100.

Best recorded AIME25 pass@8: **36.7% at step 60**. Last recorded: **30.0% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
