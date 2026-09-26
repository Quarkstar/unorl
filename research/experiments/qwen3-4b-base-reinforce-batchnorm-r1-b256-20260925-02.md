---
title: "Batch-normalized REINFORCE · SGD"
---

# Batch-normalized REINFORCE · SGD

This earlier experiment used the custom stateless SGD worker, not the later AdamW path. Its weak result cannot isolate the reward-normalization algorithm. Overlong filtering and inference correction also differ.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-reinforce-batchnorm-r1-b256-20260925-02` |
| Record | 100 steps logged / 100 planned |
| Group | Optimizer and loss confounds |
| Optimizer | SGD |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | batch_norm_reinforce |
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
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-reinforce-batchnorm-r1-b256-20260925-02.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-reinforce-batchnorm-r1-b256-20260925-02.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.7% | 13.3% |
| 20 | 1.7% | 10.0% |
| 40 | 2.1% | 10.0% |
| 60 | 2.9% | 20.0% |
| 80 | 2.9% | 20.0% |
| 100 | 2.1% | 6.7% |


Some raw evaluation dumps were truncated or malformed; aggregated logged metrics above are retained, but per-question counts for these files are unavailable:

- `exports/aime25/dumped_evals/global_step_100_evals/aime25.jsonl`

## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 4/240 | 4/30 |
| aime25 | 20 | 4/240 | 3/30 |
| aime25 | 40 | 5/240 | 3/30 |
| aime25 | 60 | 7/240 | 6/30 |
| aime25 | 80 | 7/240 | 6/30 |

## Quantitative observations

Training response correctness averaged **9.7%** over the first 10 logged updates and **9.3%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.7974 at step 1 → 0.8754 at step 100.
- Policy gradient norm: 0.05247 at step 1 → 0.05761 at step 100.
- Mean generated response tokens: 1175 at step 1 → 1108 at step 100.

Best recorded AIME25 pass@8: **20.0% at step 60**. Last recorded: **6.7% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
