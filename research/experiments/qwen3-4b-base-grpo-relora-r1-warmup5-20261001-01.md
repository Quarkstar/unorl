---
title: "GRPO · standard rank-1 ReLoRA, restart ramp"
---

# GRPO · standard rank-1 ReLoRA, restart ramp

Launched the first of two matched 100-step ReLoRA trials: standard full-layer Kaiming rank-one LoRA, alpha 32, native AdamW LR 1.5e-5, unchanged eight rollouts / 32 prompts / 8192 response budget. Merge at steps 40 and 80, clear complete Adam history, then apply LR multipliers 0, .25, .5, .75, 1 over the next five updates; initial warmup is zero. An otherwise identical constant-LR control is queued after successful completion. This isolates restart warmup with full resets, not the complete original ReLoRA recipe. Log teacher-forced KL/log-probability changes on up to 128 real response tokens per rank, token-weighted across eight ranks; accumulated low-rank spectra at merges and at step 100; per-rank allocator peaks plus ten-second NVML samples. All 83 unit/integrity tests passed; actual eight-GPU FSDP2 verified merging, scheduler behavior, exact next-update reproduction after resume, and dense export. Learning outcomes are pending.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-relora-r1-warmup5-20261001-01` |
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
| Restart warmup (updates) | 5 |
| Merge/reset interval (updates) | 40 |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | not retained |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-relora-r1-warmup5-20261001-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

No metric records survived, so no curve or score is fabricated.

## Evaluation results

No AIME25 checkpoint evaluation is retained.

## Quantitative observations


## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
