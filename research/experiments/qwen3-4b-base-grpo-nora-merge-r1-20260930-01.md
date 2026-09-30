---
title: "GRPO · full NoRA-init with merge/reset"
---

# GRPO · full NoRA-init with merge/reset

Launched a 100-step controlled follow-up to full-layer NoRA-init. The sole profile addition is a merge interval of 40 updates: merge at steps 40 and 80, fresh normalized sign A and zero B, clear adapter AdamW moments and step counters, and preserve the constant LR/scheduler. All 36 layers, rank 1 / alpha 1, eight rollouts, model/data, batch and response cap match the full NoRA-init reference. Backbone W is synchronized before the native adapter at startup/resume and merges, avoiding lost or double-counted updates. This explicit reset baseline does not preserve Adam continuity or guarantee high effective rank. Eight-GPU FSDP2 merge, continued AdamW training, checkpoint resume and dense export checks passed before launch. Learning and memory conclusions remain pending; ten-second NVML samples and exact per-update training allocator peaks are recorded locally, including merge work.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-nora-merge-r1-20260930-01` |
| Record | No retained metrics |
| Group | Controlled follow-up experiments |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | grpo |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 1 |
| LoRA initialization | nora_init |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Merge/reset interval (updates) | 40 |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | not retained |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-nora-merge-r1-20260930-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

No metric records survived, so no curve or score is fabricated.

## Evaluation results

No AIME25 checkpoint evaluation is retained.

## Quantitative observations


## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
