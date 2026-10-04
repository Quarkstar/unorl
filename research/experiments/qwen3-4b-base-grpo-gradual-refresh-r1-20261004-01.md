---
title: "GRPO · ten-increment compensated refresh, rank 1"
---

# GRPO · ten-increment compensated refresh, rank 1

100 updates from base, matched historical standard rank-one GRPO recipe: native AdamW, alpha 32, constant LR 1.5e-5, 32 prompts with eight rollouts, 8192 response tokens, eight GPUs. Replace each abrupt 20-degree refresh with ten 2-degree fixed-plane increments after updates 40-49 and 80-89. Ordinary optimizer updates occur between increments. Retain B, compensate the frozen base to preserve effective weights in exact arithmetic, preserve Adam counters and A moments, approximately project B first moments by actual row cosine, and retain B variances. This is approximate moment transport; unseen gradient history is not reconstructed. Two compressed correction directions per cycle give rank capacity at most five including the active adapter, versus at most three for the one-shot trial. Native eight-rank FSDP validation passed, including exact recovery from an in-transition checkpoint and base-weight extraction; it did not launch an inference engine. AIME25 avg@8/pass@8 every twenty updates, hourly health checks, ten-second NVML sampling, and per-update training allocator peaks. Compare with existing historical standard LoRA, completed fresh standard, and one-shot refresh; no extra baseline rerun. Outcome remains unproven.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-gradual-refresh-r1-20261004-01` |
| Record | 0 steps logged / 100 planned |
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

```{figure} ../figures/qwen3-4b-base-grpo-gradual-refresh-r1-20261004-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.7% | 10.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 4/240 | 3/30 |

## Quantitative observations


Best recorded AIME25 pass@8: **10.0% at step 0**. Last recorded: **10.0% at step 0**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).

## Gradual transition compared with retained baselines

Ten increments follow updates 40-49 and 80-89. Their first affected training updates are 41-50 and 81-90; updates 51-60 and 91-100 measure the following ten updates. Complete windows are shown below. Independent rollout trajectories and training-seed uncertainty prevent causal attribution from this single trial. Rank capacity also differs from the one-shot method.

```{figure} ../figures/comparison-gradual-refresh.svg
:alt: Historical and fresh standard LoRA, one-shot refresh, and the current ten-increment trial; missing candidate checkpoints are not extrapolated.

Historical and fresh standard LoRA, one-shot refresh, and the current ten-increment trial; missing candidate checkpoints are not extrapolated.
```

### Historical standard LoRA

| Training updates | Reference correctness | Gradual correctness | Difference |
|---|---:|---:|---:|

No complete paired training window is available yet.

Latest common AIME25 evaluation: **step 0**.

| Metric | Gradual minus reference | Question-bootstrap 95% interval |
|---|---:|---:|
| avg@8 | -0.83 pp | [-3.33, +1.25] pp |
| pass@8 | -3.33 pp | [-16.67, +10.00] pp |

[Download all metric windows, question-level intervals, and configuration differences](../data/gradual-refresh-vs-historical-standard-analysis.json).

### Completed fresh standard LoRA

| Training updates | Reference correctness | Gradual correctness | Difference |
|---|---:|---:|---:|

No complete paired training window is available yet.

Latest common AIME25 evaluation: **step 0**.

| Metric | Gradual minus reference | Question-bootstrap 95% interval |
|---|---:|---:|
| avg@8 | -1.25 pp | [-5.00, +2.08] pp |
| pass@8 | -3.33 pp | [-16.67, +10.00] pp |

[Download all metric windows, question-level intervals, and configuration differences](../data/gradual-refresh-vs-fresh-standard-analysis.json).

### Completed one-shot refresh

| Training updates | Reference correctness | Gradual correctness | Difference |
|---|---:|---:|---:|

No complete paired training window is available yet.

Latest common AIME25 evaluation: **step 0**.

| Metric | Gradual minus reference | Question-bootstrap 95% interval |
|---|---:|---:|
| avg@8 | -1.67 pp | [-3.75, +0.00] pp |
| pass@8 | -3.33 pp | [-10.00, +0.00] pp |

[Download all metric windows, question-level intervals, and configuration differences](../data/gradual-refresh-vs-one-shot-analysis.json).

Question bootstrap retains eight-response groups but does not establish equivalence or account for training-seed variance. Startup, training correctness, and intermediate evaluations do not prove the final objective.
