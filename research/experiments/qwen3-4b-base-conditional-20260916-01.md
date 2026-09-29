---
title: "Archived · conditional online SFT"
---

# Archived · conditional online SFT

Both correct and incorrect self-generated responses receive positive cross-entropy updates under different system instructions. Evaluation improves temporarily and later deteriorates. This records the retired method; its code is no longer part of the active UNORL package. The historical entropy implementation/reduction must be checked before absolute comparisons to RL entropy.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-conditional-20260916-01` |
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
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-conditional-20260916-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-conditional-20260916-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.9% | 16.7% |
| 20 | 1.2% | 6.7% |
| 40 | 1.2% | 6.7% |
| 60 | 2.9% | 16.7% |
| 80 | 7.5% | 16.7% |
| 100 | 9.2% | 23.3% |
| 120 | 12.9% | 23.3% |
| 140 | 12.9% | 30.0% |
| 160 | 11.2% | 20.0% |
| 180 | 11.7% | 23.3% |
| 200 | 12.1% | 23.3% |
| 220 | 7.9% | 23.3% |
| 240 | 5.8% | 23.3% |
| 260 | 3.3% | 13.3% |
| 280 | 0.8% | 3.3% |
| 300 | 0.0% | 0.0% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.1% | 10.0% |
| 20 | 2.5% | 10.0% |
| 40 | 1.7% | 10.0% |
| 60 | 3.8% | 16.7% |
| 80 | 8.3% | 16.7% |
| 100 | 8.3% | 23.3% |
| 120 | 10.4% | 23.3% |
| 140 | 11.2% | 23.3% |
| 160 | 10.8% | 23.3% |
| 180 | 13.8% | 30.0% |
| 200 | 12.5% | 33.3% |
| 220 | 9.6% | 26.7% |
| 240 | 6.7% | 23.3% |
| 260 | 4.6% | 10.0% |
| 280 | 2.1% | 10.0% |
| 300 | 0.8% | 3.3% |


Some raw evaluation dumps were truncated or malformed; aggregated logged metrics above are retained, but per-question counts for these files are unavailable:

- `exports/aime25/dumped_evals/global_step_40_evals/aime25.jsonl`
- `exports/aime26/dumped_evals/global_step_220_evals/aime26.jsonl`

## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 7/240 | 5/30 |
| aime25 | 20 | 3/240 | 2/30 |
| aime25 | 60 | 7/240 | 5/30 |
| aime25 | 80 | 18/240 | 5/30 |
| aime25 | 100 | 22/240 | 7/30 |
| aime25 | 120 | 31/240 | 7/30 |
| aime25 | 140 | 31/240 | 9/30 |
| aime25 | 160 | 27/240 | 6/30 |
| aime25 | 180 | 28/240 | 7/30 |
| aime25 | 200 | 29/240 | 7/30 |
| aime25 | 220 | 19/240 | 7/30 |
| aime25 | 240 | 14/240 | 7/30 |
| aime25 | 260 | 8/240 | 4/30 |
| aime25 | 280 | 2/240 | 1/30 |
| aime25 | 300 | 0/240 | 0/30 |
| aime26 | 0 | 5/240 | 3/30 |
| aime26 | 20 | 6/240 | 3/30 |
| aime26 | 40 | 4/240 | 3/30 |
| aime26 | 60 | 9/240 | 5/30 |
| aime26 | 80 | 20/240 | 5/30 |
| aime26 | 100 | 20/240 | 7/30 |
| aime26 | 120 | 25/240 | 7/30 |
| aime26 | 140 | 27/240 | 7/30 |
| aime26 | 160 | 26/240 | 7/30 |
| aime26 | 180 | 33/240 | 9/30 |
| aime26 | 200 | 30/240 | 10/30 |
| aime26 | 240 | 16/240 | 7/30 |
| aime26 | 260 | 11/240 | 3/30 |
| aime26 | 280 | 5/240 | 3/30 |
| aime26 | 300 | 2/240 | 1/30 |

## Quantitative observations

Training response correctness averaged **7.0%** over the first 10 logged updates and **2.3%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.001339 at step 101 → 0.01718 at step 300.
- Policy gradient norm: 0.3689 at step 1 → 0.4935 at step 300.
- Mean generated response tokens: 1145 at step 1 → 1276 at step 300.

Best recorded AIME25 pass@8: **30.0% at step 140**. Last recorded: **0.0% at step 300**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
