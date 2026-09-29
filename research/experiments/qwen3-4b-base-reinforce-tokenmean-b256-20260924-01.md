---
title: "Token-mean REINFORCE · SGD"
---

# Token-mean REINFORCE · SGD

Token-mean reduction removed the earlier loss-scale problem. The custom worker still used SGD, so this stopped run is not an AdamW algorithm comparison.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-reinforce-tokenmean-b256-20260924-01` |
| Record | 73 steps logged / 100 planned |
| Group | Optimizer and loss confounds |
| Optimizer | SGD |
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
| Exit status | 1 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-reinforce-tokenmean-b256-20260924-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-reinforce-tokenmean-b256-20260924-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.2% | 10.0% |
| 20 | 2.9% | 20.0% |
| 40 | 2.1% | 16.7% |
| 60 | 2.9% | 13.3% |


## Quantitative observations

Training response correctness averaged **9.1%** over the first 10 logged updates and **8.9%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.9756 at step 1 → 0.8989 at step 73.
- Policy gradient norm: 0.0769 at step 1 → 0.0843 at step 73.
- Mean generated response tokens: 1180 at step 1 → 1064 at step 73.

Best recorded AIME25 pass@8: **20.0% at step 20**. Last recorded: **13.3% at step 60**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
