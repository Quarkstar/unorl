---
title: "Mean-centered REINFORCE · SGD"
---

# Mean-centered REINFORCE · SGD

Batch-mean centering without standard-deviation scaling did not show a convincing gain before stopping. The SGD optimizer confounds comparison with the successful AdamW profiles.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-reinforce-batchmean-b256-20260924-01` |
| Record | 22 steps logged / 100 planned |
| Group | Optimizer and loss confounds |
| Optimizer | SGD |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | batch_mean_reinforce |
| Policy loss | signed_reinforce |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 32 |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Prompts × responses | 256 × 1 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | not retained |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-reinforce-batchmean-b256-20260924-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-reinforce-batchmean-b256-20260924-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.7% | 10.0% |
| 20 | 1.7% | 13.3% |


Some raw evaluation dumps were truncated or malformed; aggregated logged metrics above are retained, but per-question counts for these files are unavailable:

- `exports/aime25/dumped_evals/global_step_20_evals/aime25.jsonl`

## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 4/240 | 3/30 |

## Quantitative observations

Training response correctness averaged **8.9%** over the first 10 logged updates and **9.8%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8489 at step 1 → 0.8407 at step 22.
- Policy gradient norm: 0.02858 at step 1 → 0.03039 at step 22.
- Mean generated response tokens: 1245 at step 1 → 1081 at step 22.

Best recorded AIME25 pass@8: **13.3% at step 20**. Last recorded: **13.3% at step 20**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
