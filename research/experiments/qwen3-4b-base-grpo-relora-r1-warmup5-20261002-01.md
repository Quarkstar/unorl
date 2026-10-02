---
title: "GRPO · standard rank-1 ReLoRA, restart ramp (fixed probe)"
---

# GRPO · standard rank-1 ReLoRA, restart ramp (fixed probe)

Restarted from the original base model after correcting the trajectory-prefix diagnostic. Recipe unchanged: standard Kaiming rank-one LoRA, alpha 32, native AdamW LR 1.5e-5, eight rollouts, 32 prompts/update, 8192 response tokens, 100 steps. Merge/reset at steps 40/80, then five-update LR multipliers 0, .25, .5, .75, 1; no initial warmup. Constant-LR control queued after successful completion. Regression tests use native SkyRL left-padding and response-slice construction. Actual eight-GPU tests now exercise the real worker merge method, collective KL aggregation and accumulated-rank diagnostics, plus exact next-update reproduction after checkpoint loading and dense export. All 85 tests passed. The previous attempt failed before any merge and therefore does not measure ReLoRA behavior. Logs/evaluation dumps from that attempt are retained. Outcomes for this restarted comparison are pending.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-relora-r1-warmup5-20261002-01` |
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

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-relora-r1-warmup5-20261002-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

No metric records survived, so no curve or score is fabricated.

## Evaluation results

No AIME25 checkpoint evaluation is retained.

## Quantitative observations


## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
