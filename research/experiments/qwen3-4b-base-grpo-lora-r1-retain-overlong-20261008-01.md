---
title: "GRPO · retain-overlong rank-1 LoRA (filtering disabled)"
---

# GRPO · retain-overlong rank-1 LoRA (filtering disabled)

Standard rank-one LoRA GRPO from base, identical to the historical baseline except generator.apply_overlong_filtering is false, so truncated responses keep their loss mask and train on their negative reward. Qwen3-4B-Base, all-linear rank 1 / alpha 32, native AdamW constant LR 1.5e-5, no warmup, GRPO without group-std normalization or KL, token-mean loss, no policy clipping, token TIS correction with clip 1e9, 32 prompts with eight rollouts, 8192 response budget, seed 42, eight GPUs, 100 updates, AIME25 avg@8/pass@8 every 20 updates. Isolates the retain-versus-filter choice relative to the post-step-60 plateau in which training reward flattens while response length grows and the truncation rate rises. Single run on 30 questions; no endpoint ranking is assumed. Metrics and figures are added from this snapshot once all updates and evaluations complete.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-lora-r1-retain-overlong-20261008-01` |
| Record | 100 steps logged / 100 planned |
| Group | Controlled follow-up experiments |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | grpo |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 32 |
| LoRA initialization | kaiming (default) |
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

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-lora-r1-retain-overlong-20261008-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-lora-r1-retain-overlong-20261008-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.7% | 10.0% |
| 20 | 5.8% | 20.0% |
| 40 | 10.8% | 26.7% |
| 60 | 15.8% | 30.0% |
| 80 | 15.8% | 30.0% |
| 100 | 17.1% | 33.3% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 4/240 | 3/30 |
| aime25 | 20 | 14/240 | 6/30 |
| aime25 | 40 | 26/240 | 8/30 |
| aime25 | 60 | 38/240 | 9/30 |
| aime25 | 80 | 38/240 | 9/30 |
| aime25 | 100 | 41/240 | 10/30 |

## Quantitative observations

Training response correctness averaged **9.6%** over the first 10 logged updates and **36.4%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8764 at step 1 → 0.1413 at step 100.
- Policy gradient norm: 0.03117 at step 1 → 0.02769 at step 100.
- Mean generated response tokens: 1166 at step 1 → 3663 at step 100.

Best recorded AIME25 pass@8: **33.3% at step 100**. Last recorded: **33.3% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
