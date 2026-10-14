---
title: "Raw signed REINFORCE · rank 1 alpha 1 lr 1e-5"
---

# Raw signed REINFORCE · rank 1 alpha 1 lr 1e-5

Matched estimator comparison arm: raw signed REINFORCE on the terminal +1/-1 reward with no centering or whitening (unorl signed_reinforce estimator and loss), 256 prompts x 1, rank-1 LoRA alpha 1, LR 1e-5, token_mean. Same batch, response budget and evaluation as the GRPO and REINFORCE++ arms.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-reinforce-adamw-r1-a1-20261013-01` |
| Record | 100 steps logged / 100 planned |
| Group | Controlled follow-up experiments |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | signed_reinforce |
| Policy loss | signed_reinforce |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 1 |
| LoRA initialization | kaiming (default) |
| Learning rate | 1e-05 |
| Warmup steps | 0 |
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 256 × 1 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-reinforce-adamw-r1-a1-20261013-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-reinforce-adamw-r1-a1-20261013-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.2% | 10.0% |
| 20 | 1.7% | 10.0% |
| 40 | 2.5% | 16.7% |
| 60 | 4.6% | 23.3% |
| 80 | 2.9% | 13.3% |
| 100 | 2.9% | 23.3% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.5% | 10.0% |
| 20 | 2.5% | 13.3% |
| 40 | 3.3% | 13.3% |
| 60 | 4.6% | 23.3% |
| 80 | 1.2% | 6.7% |
| 100 | 2.1% | 6.7% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 3/240 | 3/30 |
| aime25 | 20 | 4/240 | 3/30 |
| aime25 | 40 | 6/240 | 5/30 |
| aime25 | 60 | 11/240 | 7/30 |
| aime25 | 80 | 7/240 | 4/30 |
| aime25 | 100 | 7/240 | 7/30 |
| aime26 | 0 | 6/240 | 3/30 |
| aime26 | 20 | 6/240 | 4/30 |
| aime26 | 40 | 8/240 | 4/30 |
| aime26 | 60 | 11/240 | 7/30 |
| aime26 | 80 | 3/240 | 2/30 |
| aime26 | 100 | 5/240 | 2/30 |

## Quantitative observations

Training response correctness averaged **9.0%** over the first 10 logged updates and **9.0%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8183 at step 1 → 0.779 at step 100.
- Policy gradient norm: 0.002547 at step 1 → 0.003138 at step 100.
- Mean generated response tokens: 1170 at step 1 → 1135 at step 100.

Best recorded AIME25 pass@8: **23.3% at step 60**. Last recorded: **23.3% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
