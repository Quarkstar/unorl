---
title: "GRPO · ten-increment compensated refresh, rank 1"
---

# GRPO · ten-increment compensated refresh, rank 1

100 updates from base, matched historical standard rank-one GRPO recipe: native AdamW, alpha 32, constant LR 1.5e-5, 32 prompts with eight rollouts, 8192 response tokens, eight GPUs. Replace each abrupt 20-degree refresh with ten 2-degree fixed-plane increments after updates 40-49 and 80-89. Ordinary optimizer updates occur between increments. Retain B, compensate the frozen base to preserve effective weights in exact arithmetic, preserve Adam counters and A moments, approximately project B first moments by actual row cosine, and retain B variances. This is approximate moment transport; unseen gradient history is not reconstructed. Two compressed correction directions per cycle give rank capacity at most five including the active adapter, versus at most three for the one-shot trial. Native eight-rank FSDP validation passed, including exact recovery from an in-transition checkpoint and base-weight extraction; it did not launch an inference engine. AIME25 avg@8/pass@8 every twenty updates, hourly health checks, ten-second NVML sampling, and per-update training allocator peaks. Compare with existing historical standard LoRA, completed fresh standard, and one-shot refresh; no extra baseline rerun. Outcome remains unproven.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-gradual-refresh-r1-20261004-01` |
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

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-gradual-refresh-r1-20261004-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

No metric records survived, so no curve or score is fabricated.

## Evaluation results

No AIME25 checkpoint evaluation is retained.

## Quantitative observations


## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
