---
title: "GRPO · prepared-history rank-aware ReLoRA, rank 1"
---

# GRPO · prepared-history rank-aware ReLoRA, rank 1

100 updates from base using the historical standard rank-one LoRA GRPO recipe: Qwen3-4B-Base, all-linear rank 1 / alpha 32, native AdamW, constant LR 1.5e-5, 32 prompts with eight rollouts, response budget 8192 and eight GPUs. Prepare fixed two-direction input/output gradient history during updates 21-40 and 61-80; reduce full-update projections before clipping and cross moments. After 40/80 choose norm-matched new factors by observed normal-gradient descent, constrained to retain at least 90% of each side's best local total-descent score on a 256-angle grid. Compensate the frozen base, install observed-window Adam moments/counters and preserve the global constant scheduler. The constraint is not a baseline accuracy bound; the mean-gradient reference is fixed at preparation and its drift is logged. Refresh bases each cycle, measure actual accumulated rank, and skip layers without positive observed normal-descent benefit. Native eight-rank GRPO worker tests passed active/closed checkpoint replay, two transfers, gradient collection and dense export. Live inference, full-model peak VRAM and reward/performance remain to be established by this trial. AIME25 avg@8/pass@8 every 20 updates; hourly checks, 10-second NVML sampling and per-update training allocator peaks. Reuse existing standard LoRA and completed failed refresh trials; no extra baseline or budget extension. Outcome remains unproven until observed results.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-prepared-r1-20261004-01` |
| Record | 100 steps logged / 100 planned |
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
| Exit status | 0 |
| Merges enabled | True |
| First merge global step | 40 |
| Refresh angle (degrees) | None |
| Resume checkpoint | None |
| Preparation window (updates) | 20 |
| Minimum observed local descent fraction | 0.9 |
| Direction angles per factor | 256 |
| Adam transferred-history counter | window observations; global scheduler unchanged |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-prepared-r1-20261004-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-prepared-r1-20261004-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 3.3% | 16.7% |
| 20 | 7.9% | 26.7% |
| 40 | 15.0% | 40.0% |
| 60 | 14.6% | 26.7% |
| 80 | 17.9% | 36.7% |
| 100 | 15.8% | 26.7% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 8/240 | 5/30 |
| aime25 | 20 | 19/240 | 8/30 |
| aime25 | 40 | 36/240 | 12/30 |
| aime25 | 60 | 35/240 | 8/30 |
| aime25 | 80 | 43/240 | 11/30 |
| aime25 | 100 | 38/240 | 8/30 |

## Quantitative observations

Training response correctness averaged **9.3%** over the first 10 logged updates and **33.4%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8799 at step 1 → 0.1239 at step 100.
- Policy gradient norm: 0.03134 at step 1 → 0.0333 at step 100.
- Mean generated response tokens: 1455 at step 1 → 4047 at step 100.

Best recorded AIME25 pass@8: **40.0% at step 40**. Last recorded: **26.7% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).

## Independently audited prefix through step 100

This is a partial-run checkpoint audit, not a completed experiment or proof of performance parity. All eight saved optimizer states, scheduler/protocol, finite local moments, preparation state, raw evaluation groups, and the recorded training-memory prefix were checked.

Training allocator peak so far: **17.826 GiB allocated / 18.762 GiB reserved**, from **800 worker-update records**. These are policy allocator measurements, not total-device memory.

[Download the checkpoint audit and source hashes](../data/qwen3-4b-base-grpo-prepared-r1-20261004-01-checkpoint-step100.json).

Actual saved-factor mean stable rank: **1.4752**. Energy-weighted fraction outside each matrix's leading singular direction: **29.83%**. This measures the accumulated factor update, excluding base-rounding residuals; it is distinct from normalized direction-span rank.

[Download saved-factor diagnostics and source hashes](../data/qwen3-4b-base-grpo-prepared-r1-20261004-01-factor-span-step100.json).

```{figure} ../figures/prepared-update-geometry.svg
:alt: Recorded optimizer updates and boundary drift. Squares on the rank panel are independent saved-factor measurements at the indicated steps; missing ranks are not interpolated.

Recorded optimizer updates and boundary drift. Squares on the rank panel are independent saved-factor measurements at the indicated steps; missing ranks are not interpolated.
```
The [mathematical and checkpoint analysis](../notes/lora-training-2025-2026.md) explains finite-window Adam history, switch continuity, and rank-growth limitations. [Download the matched paired-question comparison](../data/prepared-base-comparison-analysis.json). AIME's 30 questions and a single training trajectory do not establish equivalence.
