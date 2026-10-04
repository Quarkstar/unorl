---
title: "GRPO · fresh standard rank-1 LoRA control (100 steps from base)"
---

# GRPO · fresh standard rank-1 LoRA control (100 steps from base)

Fresh 100-update standard LoRA control for the completed gradual-refresh trial. Same refresh worker, model, dataset, initial adapter, AdamW, constant LR 1.5e-5, alpha 32, eight rollouts, 32 prompts/update, 8192 response budget, eight GPUs, checkpoint/evaluation cadence and telemetry. Only method flags differ: refresh disabled and angle zero. No resumed checkpoint and no merge or optimizer-state reset. AIME25 sampled avg@8/pass@8 every twenty updates. Hourly monitoring and ten-second NVML sampling, plus actual training allocator peaks on each rank. This tests the original budget and runtime rather than extending the candidate. Performance parity remains unproven until complete held-out results and replication; no outcome is assumed at launch.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-standard-r1-refresh-control-20261003-01` |
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
| Merges enabled | False |
| First merge global step | 40 |
| Refresh angle (degrees) | 0.0 |
| Resume checkpoint | None |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-standard-r1-refresh-control-20261003-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-standard-r1-refresh-control-20261003-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.9% | 13.3% |
| 20 | 5.4% | 23.3% |
| 40 | 12.1% | 23.3% |
| 60 | 12.9% | 26.7% |
| 80 | 16.7% | 36.7% |
| 100 | 17.5% | 36.7% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 7/240 | 4/30 |
| aime25 | 20 | 13/240 | 7/30 |
| aime25 | 40 | 29/240 | 7/30 |
| aime25 | 60 | 31/240 | 8/30 |
| aime25 | 80 | 40/240 | 11/30 |
| aime25 | 100 | 42/240 | 11/30 |

## Quantitative observations

Training response correctness averaged **10.0%** over the first 10 logged updates and **35.7%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.7918 at step 1 → 0.1277 at step 100.
- Policy gradient norm: 0.0311 at step 1 → 0.03228 at step 100.
- Mean generated response tokens: 1397 at step 1 → 4032 at step 100.

Best recorded AIME25 pass@8: **36.7% at step 80**. Last recorded: **36.7% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).

## Matched comparison with the fresh standard run

The additional fresh standard run checks the current implementation with refresh disabled; the historical standard-LoRA baseline remains valid. These independently diverged runs do not isolate the causal effect of rotation. The refresh candidate uses one 20-degree rotation at each boundary. The ten-increment variant is reported separately in its [completed experiment page](qwen3-4b-base-grpo-gradual-refresh-r1-20261004-01.md).

```{figure} ../figures/comparison-refresh-base-fresh.svg
:alt: Standard LoRA versus one-shot compensated refresh, both starting from base with a 100-update budget. A partial standard run is not a final endpoint comparison.

Standard LoRA versus one-shot compensated refresh, both starting from base with a 100-update budget. A partial standard run is not a final endpoint comparison.
```

### Training correctness in complete windows

| Updates | Standard LoRA | One-shot refresh | Refresh − standard |
|---|---:|---:|---:|
| 1-20 | 12.21% | 12.07% | -0.14 pp |
| 21-40 | 21.37% | 24.71% | +3.34 pp |
| 41-45 | 31.41% | 38.91% | +7.50 pp |
| 46-60 | 35.44% | 37.86% | +2.42 pp |
| 61-80 | 38.44% | 38.46% | +0.02 pp |
| 81-100 | 38.48% | 38.63% | +0.16 pp |

### Latest paired-question analysis: step 100

| Metric | Refresh − standard | Question-bootstrap 95% interval |
|---|---:|---:|
| avg@8 | -0.42 pp | [-3.33, +2.50] pp |
| avg@8: improvement from step 0 | -0.83 pp | [-5.00, +3.33] pp |
| pass@8 | +0.00 pp | [-16.67, +16.67] pp |
| pass@8: improvement from step 0 | +0.00 pp | [-20.00, +23.33] pp |

Intervals resample the 30 whole questions, retaining each group of eight responses; they do not measure training-seed uncertainty or establish equivalence. A lead before the first rotation must not be credited to the method. See the [detailed first-principles investigation](../notes/lora-training-2025-2026.md) and [download the paired analysis](../data/refresh-base-fresh-control-analysis.json).

## Completed-run protocol and training allocator

Successful exit and **100 updates** were audited against the matched recipe. Raw AIME25 groups agree with logged scores at every configured evaluation, and all **800 memory records across 8 ranks** match the update and refresh schedule. This verifies the recorded protocol, not performance parity.

| Measurement | GiB | Step | Rank |
|---|---:|---:|---:|
| Maximum training allocation | 17.939 | 76 | 5 |
| Maximum training reservation | 18.789 | 76 | 5 |

These are policy-process allocator peaks in the worker's declared training window. They are not total GPU memory or a single-device/phone estimate; inference, weight synchronization, and export are outside this measurement. [Download the completion audit and source hashes](../data/qwen3-4b-base-grpo-standard-r1-refresh-control-20261003-01-completion-audit.json).
