---
title: "GRPO · fresh standard rank-1 LoRA control (100 steps from base)"
---

# GRPO · fresh standard rank-1 LoRA control (100 steps from base)

Fresh 100-update standard LoRA control for the completed gradual-refresh trial. Same refresh worker, model, dataset, initial adapter, AdamW, constant LR 1.5e-5, alpha 32, eight rollouts, 32 prompts/update, 8192 response budget, eight GPUs, checkpoint/evaluation cadence and telemetry. Only method flags differ: refresh disabled and angle zero. No resumed checkpoint and no merge or optimizer-state reset. AIME25 sampled avg@8/pass@8 every twenty updates. Hourly monitoring and ten-second NVML sampling, plus actual training allocator peaks on each rank. This tests the original budget and runtime rather than extending the candidate. Performance parity remains unproven until complete held-out results and replication; no outcome is assumed at launch.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-standard-r1-refresh-control-20261003-01` |
| Record | No retained metrics |
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

No metric records survived, so no curve or score is fabricated.

## Evaluation results

No AIME25 checkpoint evaluation is retained.

## Quantitative observations


## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
