---
title: "Archived · positive-only online SFT"
---

# Archived · positive-only online SFT

Only verified correct self-generated responses are trained. This run demonstrates learning with rejection sampling. Training correctness and prompt-level training pass@8 are distinct quantities; compare response correctness with held-out avg@8, not training pass@8 with held-out pass@1.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-positive-only-20260918-01` |
| Record | 300 steps logged / 300 planned |
| Group | Earlier research |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | grpo |
| Policy loss | cross_entropy |
| Loss reduction | token_mean |
| LoRA rank | 0 |
| LoRA alpha | — |
| LoRA initialization | kaiming (default) |
| Learning rate | 1e-06 |
| Warmup steps | 5 |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-positive-only-20260918-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-positive-only-20260918-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 0.8% | 6.7% |
| 20 | 5.4% | 20.0% |
| 40 | 11.7% | 26.7% |
| 60 | 13.3% | 30.0% |
| 80 | 13.8% | 33.3% |
| 100 | 15.8% | 33.3% |
| 120 | 19.6% | 33.3% |
| 140 | 14.6% | 36.7% |
| 160 | 15.4% | 36.7% |
| 180 | 15.4% | 33.3% |
| 200 | 17.5% | 33.3% |
| 220 | 20.8% | 33.3% |
| 240 | 15.0% | 26.7% |
| 260 | 15.4% | 36.7% |
| 280 | 19.6% | 40.0% |
| 300 | 15.4% | 30.0% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.7% | 13.3% |
| 20 | 5.4% | 20.0% |
| 40 | 10.8% | 33.3% |
| 60 | 14.6% | 30.0% |
| 80 | 14.6% | 26.7% |
| 100 | 16.2% | 30.0% |
| 120 | 15.0% | 33.3% |
| 140 | 13.3% | 23.3% |
| 160 | 15.0% | 26.7% |
| 180 | 15.8% | 30.0% |
| 200 | 17.5% | 30.0% |
| 220 | 19.2% | 43.3% |
| 240 | 14.6% | 26.7% |
| 260 | 13.8% | 26.7% |
| 280 | 14.2% | 30.0% |
| 300 | 15.0% | 30.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 2/240 | 2/30 |
| aime25 | 20 | 13/240 | 6/30 |
| aime25 | 40 | 28/240 | 8/30 |
| aime25 | 60 | 32/240 | 9/30 |
| aime25 | 80 | 33/240 | 10/30 |
| aime25 | 100 | 38/240 | 10/30 |
| aime25 | 120 | 47/240 | 10/30 |
| aime25 | 140 | 35/240 | 11/30 |
| aime25 | 160 | 37/240 | 11/30 |
| aime25 | 180 | 37/240 | 10/30 |
| aime25 | 200 | 42/240 | 10/30 |
| aime25 | 220 | 50/240 | 10/30 |
| aime25 | 240 | 36/240 | 8/30 |
| aime25 | 260 | 37/240 | 11/30 |
| aime25 | 280 | 47/240 | 12/30 |
| aime25 | 300 | 37/240 | 9/30 |
| aime26 | 0 | 4/240 | 4/30 |
| aime26 | 20 | 13/240 | 6/30 |
| aime26 | 40 | 26/240 | 10/30 |
| aime26 | 60 | 35/240 | 9/30 |
| aime26 | 80 | 35/240 | 8/30 |
| aime26 | 100 | 39/240 | 9/30 |
| aime26 | 120 | 36/240 | 10/30 |
| aime26 | 140 | 32/240 | 7/30 |
| aime26 | 160 | 36/240 | 8/30 |
| aime26 | 180 | 38/240 | 9/30 |
| aime26 | 200 | 42/240 | 9/30 |
| aime26 | 220 | 46/240 | 13/30 |
| aime26 | 240 | 35/240 | 8/30 |
| aime26 | 260 | 33/240 | 8/30 |
| aime26 | 280 | 34/240 | 9/30 |
| aime26 | 300 | 36/240 | 9/30 |

## Quantitative observations

Training response correctness averaged **7.1%** over the first 10 logged updates and **42.2%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.07736 at step 1 → 0.09549 at step 300.
- Policy gradient norm: 0.9203 at step 1 → 0.2366 at step 300.
- Mean generated response tokens: 1292 at step 1 → 4462 at step 300.

Best recorded AIME25 pass@8: **40.0% at step 280**. Last recorded: **30.0% at step 300**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
