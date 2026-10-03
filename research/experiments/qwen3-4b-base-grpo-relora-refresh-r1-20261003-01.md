---
title: "GRPO · compensated gradual refresh, rank 1 (100 steps from base)"
---

# GRPO · compensated gradual refresh, rank 1 (100 steps from base)

Main-budget trial of the proposed continuity-preserving refresh algorithm. Start from Qwen3-4B-Base, not a trained checkpoint; train exactly 100 updates with the standard rank-one LoRA GRPO recipe. Refresh after updates 40 and 80: rotate A by 20 degrees, retain B, compensate the frozen backbone, retain Adam counters and A moments, project B first moments by cosine and retain B variances. Moment transport is approximate, not a convergence guarantee. Native AdamW LR 1.5e-5 constant, alpha 32, eight rollouts, 32 prompts per update, 8192 response tokens, all eight GPUs. AIME25 avg@8 and pass@8 every twenty updates; hourly monitoring and ten-second VRAM sampling. Compare the complete 0-100 reward/evaluation curve with the historical standard rank-one reference and cold-reset trials, particularly steps 46-60. No result or superiority is assumed before completion.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-relora-refresh-r1-20261003-01` |
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
| Merges enabled | True |
| First merge global step | 40 |
| Refresh angle (degrees) | 20.0 |
| Resume checkpoint | None |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-relora-refresh-r1-20261003-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

No metric records survived, so no curve or score is fabricated.

## Evaluation results

No AIME25 checkpoint evaluation is retained.

## Quantitative observations


## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
