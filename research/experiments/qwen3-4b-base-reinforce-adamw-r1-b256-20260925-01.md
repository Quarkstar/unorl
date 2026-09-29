---
title: "Vanilla REINFORCE · AdamW"
---

# Vanilla REINFORCE · AdamW

The corrected AdamW run learns: training correctness and sampled AIME accuracy increase. Its final AIME25 results are below the rank-1 GRPO reference. Earlier SGD failures do not establish that single-rollout REINFORCE fails.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-reinforce-adamw-r1-b256-20260925-01` |
| Record | 100 steps logged / 100 planned |
| Group | Current algorithm comparison |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | signed_reinforce |
| Policy loss | signed_reinforce |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 32 |
| LoRA initialization | kaiming (default) |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Prompts × responses | 256 × 1 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-reinforce-adamw-r1-b256-20260925-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-reinforce-adamw-r1-b256-20260925-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 3.3% | 23.3% |
| 20 | 5.4% | 20.0% |
| 40 | 5.0% | 26.7% |
| 60 | 12.5% | 26.7% |
| 80 | 13.3% | 30.0% |
| 100 | 14.6% | 26.7% |


Some raw evaluation dumps were truncated or malformed; aggregated logged metrics above are retained, but per-question counts for these files are unavailable:

- `exports/aime25/dumped_evals/global_step_20_evals/aime25.jsonl`

## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 8/240 | 7/30 |
| aime25 | 40 | 12/240 | 8/30 |
| aime25 | 60 | 30/240 | 8/30 |
| aime25 | 80 | 32/240 | 9/30 |
| aime25 | 100 | 35/240 | 8/30 |

## Quantitative observations

Training response correctness averaged **8.7%** over the first 10 logged updates and **37.3%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8637 at step 1 → 0.1304 at step 100.
- Policy gradient norm: 0.08241 at step 1 → 0.05507 at step 100.
- Mean generated response tokens: 1285 at step 1 → 4069 at step 100.

Best recorded AIME25 pass@8: **30.0% at step 80**. Last recorded: **26.7% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
