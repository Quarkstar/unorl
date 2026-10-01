---
title: "Rank-1 LoRA GRPO"
---

# Rank-1 LoRA GRPO

Rank-1 LoRA improves over initialization with the adjusted learning rate. It uses 32 prompts × 8 responses per update; the single-rollout experiments use 256 prompts × 1 response. Response count is matched, but prompt exposure, token count and wall time are not.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-lora-r1-blog-20260923-01` |
| Record | 100 steps logged / 100 planned |
| Group | Current algorithm comparison |
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

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-lora-r1-blog-20260923-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-lora-r1-blog-20260923-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.5% | 13.3% |
| 20 | 4.2% | 23.3% |
| 40 | 11.2% | 26.7% |
| 60 | 15.4% | 30.0% |
| 80 | 13.8% | 30.0% |
| 100 | 20.0% | 43.3% |


Some raw evaluation dumps were truncated or malformed; aggregated logged metrics above are retained, but per-question counts for these files are unavailable:

- `exports/aime25/dumped_evals/global_step_0_evals/aime25.jsonl`

## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 20 | 10/240 | 7/30 |
| aime25 | 40 | 27/240 | 8/30 |
| aime25 | 60 | 37/240 | 9/30 |
| aime25 | 80 | 33/240 | 9/30 |
| aime25 | 100 | 48/240 | 13/30 |

## Quantitative observations

Training response correctness averaged **9.6%** over the first 10 logged updates and **34.9%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.7127 at step 1 → 0.1242 at step 100.
- Policy gradient norm: 0.02638 at step 1 → 0.03516 at step 100.
- Mean generated response tokens: 1172 at step 1 → 4006 at step 100.

Best recorded AIME25 pass@8: **43.3% at step 100**. Last recorded: **43.3% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
