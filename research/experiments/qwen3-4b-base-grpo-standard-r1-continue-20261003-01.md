---
title: "GRPO · standard rank-1 LoRA continuation (shared step-100 checkpoint)"
---

# GRPO · standard rank-1 LoRA continuation (shared step-100 checkpoint)

Matched control for the gradual-refresh candidate. Load the historical standard rank-one LoRA global-step-100 model, native AdamW moments/counters, training RNG shards and dataloader position. Train 100 additional updates, global steps 101–200. Disable all merges and refreshes; use the same worker, native optimizer, scheduler, rollout synchronization, monitoring and diagnostic plumbing as the candidate. Rank one / alpha 32, LR 1.5e-5, eight rollouts, 32 prompts/update, 8192 response tokens, no KL, no warmup. AIME25 sampled avg@8/pass@8 every 20 steps. Save resumable checkpoints every 20 steps, retaining the newest; preserve the shared source checkpoint. Initial evaluation is at global step 100. Fresh inference engines are initialized with matching configuration, but their historical sampling RNG is not restored, so sampling is not bitwise continuation. Hourly process/metric checks and ten-second NVML samples are enabled. Curves and tables below reflect the latest retained metrics; outcome analysis follows completion. This is an advanced-policy continuation, not a new 0–100 trial.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-standard-r1-continue-20261003-01` |
| Record | No retained metrics |
| Group | Matched continuation from the standard-LoRA step-100 checkpoint |
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

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-standard-r1-continue-20261003-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

No metric records survived, so no curve or score is fabricated.

## Evaluation results

No AIME25 checkpoint evaluation is retained.

## Quantitative observations


## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
