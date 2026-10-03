---
title: "GRPO · fresh standard rank-1 LoRA control (100 steps from base)"
---

# GRPO · fresh standard rank-1 LoRA control (100 steps from base)

Fresh 100-update standard LoRA control for the completed gradual-refresh trial. Same refresh worker, model, dataset, initial adapter, AdamW, constant LR 1.5e-5, alpha 32, eight rollouts, 32 prompts/update, 8192 response budget, eight GPUs, checkpoint/evaluation cadence and telemetry. Only method flags differ: refresh disabled and angle zero. No resumed checkpoint and no merge or optimizer-state reset. AIME25 sampled avg@8/pass@8 every twenty updates. Hourly monitoring and ten-second NVML sampling, plus actual training allocator peaks on each rank. This tests the original budget and runtime rather than extending the candidate. Performance parity remains unproven until complete held-out results and replication; no outcome is assumed at launch.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-standard-r1-refresh-control-20261003-01` |
| Record | 40 steps logged / 100 planned |
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
| Restart warmup (updates) | 0 |
| Merge/reset interval (updates) | 40 |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | not retained |
| Merges enabled | False |
| First merge global step | 40 |
| Refresh angle (degrees) | 0.0 |
| Resume checkpoint | None |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-standard-r1-refresh-control-20261003-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-standard-r1-refresh-control-20261003-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.9% | 13.3% |
| 20 | 5.4% | 23.3% |
| 40 | 12.1% | 23.3% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 7/240 | 4/30 |
| aime25 | 20 | 13/240 | 7/30 |
| aime25 | 40 | 29/240 | 7/30 |

## Quantitative observations

Training response correctness averaged **10.0%** over the first 10 logged updates and **24.5%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.7918 at step 1 → 0.2079 at step 40.
- Policy gradient norm: 0.0311 at step 1 → 0.03798 at step 40.
- Mean generated response tokens: 1397 at step 1 → 2090 at step 40.

Best recorded AIME25 pass@8: **23.3% at step 20**. Last recorded: **23.3% at step 40**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
