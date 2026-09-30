---
title: "Single-GPU LoRA launch record"
---

# Single-GPU LoRA launch record

Configuration is retained but there are no metrics. No learning or evaluation conclusion is possible.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-lora-r1-gpu0-20260923-01` |
| Record | No retained metrics |
| Group | Incomplete launch records |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | grpo |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 16 |
| LoRA initialization | kaiming (default) |
| Learning rate | 1e-06 |
| Warmup steps | 5 |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 4 × 8 = 32 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 1 |
| KL loss / reward | False / False |
| Exit status | stopped |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-lora-r1-gpu0-20260923-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

No metric records survived, so no curve or score is fabricated.

## Evaluation results

No AIME25 checkpoint evaluation is retained.

## Quantitative observations


## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
