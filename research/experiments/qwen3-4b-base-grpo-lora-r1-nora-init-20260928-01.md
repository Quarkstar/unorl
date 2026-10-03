---
title: "GRPO · rank-1 NoRA-init"
---

# GRPO · rank-1 NoRA-init

Completed 100 steps successfully. The main positive signal is faster early learning: training correctness averages 18.4% versus 12.2% in steps 1–20, and 33.8% versus 23.7% in steps 21–40. AIME25 avg@8 at step 20 is 9.2% versus 4.2% for standard rank-1 LoRA. Training correctness later levels off near 38–39%; this does not establish that NoRA causes the plateau. Final avg@8 / pass@8 is 17.9% / 36.7%, versus the reference's 20.0% / 43.3%. The trial changes initialization and alpha together (normalized A with alpha 1 versus Kaiming A with alpha 32), so the acceleration cannot be attributed to initialization alone. The next hypothesis is that periodic merge/reset may retain the faster learning while expanding the accumulated update beyond one fixed rank-one adapter. No merge/reset was used in this trial, and that hypothesis remains untested under this recipe.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-lora-r1-nora-init-20260928-01` |
| Record | 100 steps logged / 100 planned |
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
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-lora-r1-nora-init-20260928-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-lora-r1-nora-init-20260928-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 0.8% | 6.7% |
| 20 | 9.2% | 23.3% |
| 40 | 13.3% | 30.0% |
| 60 | 15.8% | 30.0% |
| 80 | 15.4% | 30.0% |
| 100 | 17.9% | 36.7% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 2/240 | 2/30 |
| aime25 | 20 | 22/240 | 7/30 |
| aime25 | 40 | 32/240 | 9/30 |
| aime25 | 60 | 38/240 | 9/30 |
| aime25 | 80 | 37/240 | 9/30 |
| aime25 | 100 | 43/240 | 11/30 |

## Quantitative observations

Training response correctness averaged **12.1%** over the first 10 logged updates and **35.9%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.852 at step 1 → 0.104 at step 100.
- Policy gradient norm: 0.1255 at step 1 → 0.104 at step 100.
- Mean generated response tokens: 1230 at step 1 → 4856 at step 100.

Best recorded AIME25 pass@8: **36.7% at step 100**. Last recorded: **36.7% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).

## Faster early learning: comparison with standard LoRA

Reference: [full-layer rank-1 GRPO](qwen3-4b-base-grpo-lora-r1-blog-20260923-01.md). The model, data, prompt format, AdamW LR, batch, rollout count and response cap match. Initialization and alpha change together; this is a method-and-scaling comparison.

```{figure} ../figures/comparison-nora.svg
:alt: NoRA-init versus standard full-layer LoRA. Same number of responses per update; longer responses mean token compute is not matched.

NoRA-init versus standard full-layer LoRA. Same number of responses per update; longer responses mean token compute is not matched.
```

### Training correctness by 20-step window

| Steps | NoRA-init | Standard LoRA | Difference |
|---|---:|---:|---:|
| 1–20 | 18.4% | 12.2% | +6.2 pp |
| 21–40 | 33.8% | 23.7% | +10.1 pp |
| 41–60 | 39.4% | 36.3% | +3.1 pp |
| 61–80 | 38.3% | 39.3% | -0.9 pp |
| 81–100 | 38.5% | 37.5% | +1.1 pp |

### AIME25 checkpoint comparison

| Step | NoRA avg@8 | LoRA avg@8 | NoRA pass@8 | LoRA pass@8 |
|---:|---:|---:|---:|---:|
| 0 | 0.8% | 2.5% | 6.7% | 13.3% |
| 20 | 9.2% | 4.2% | 23.3% | 23.3% |
| 40 | 13.3% | 11.2% | 30.0% | 26.7% |
| 60 | 15.8% | 15.4% | 30.0% | 30.0% |
| 80 | 15.4% | 13.8% | 30.0% | 30.0% |
| 100 | 17.9% | 20.0% | 36.7% | 43.3% |

### Interpretation and next hypothesis

The early acceleration is a promising result even though the final evaluation does not beat the reference. Step-zero scores differ despite B=0 and unchanged initial logits; these are stochastic evaluation draws, not different starting weights. NoRA's positive signal is faster improvement in both training correctness and early held-out evaluation.

The later plateau is an observation, not evidence that NoRA is its cause. A fixed rank-one update is one possible constraint; data difficulty, truncation and optimization dynamics are alternative explanations. This run does not identify the cause.

**Next hypothesis: periodic merge/reset.** Merge the learned BA update into the backbone, then initialize a fresh adapter with NoRA-init and B=0. This preserves the policy at the reset boundary in exact arithmetic while allowing the accumulated update across cycles to exceed rank one. Test whether the early learning speed returns after reset and whether later reward and AIME25 accuracy improve. Optimizer-state reset and the merge interval must be explicit experimental settings. Existing exploratory merge runs also changed the optimizer and algorithm, so they do not validate this hypothesis. The [full-layer merge/reset follow-up](qwen3-4b-base-grpo-nora-merge-r1-20260930-01.md) completed with merges at updates 40 and 80 and an explicit Adam-state reset. It did not sustain further reward improvement after resets; final AIME25 avg@8 was 15.0%.
