---
title: "GRPO · ten-increment compensated refresh, rank 1"
---

# GRPO · ten-increment compensated refresh, rank 1

100 updates from base, matched historical standard rank-one GRPO recipe: native AdamW, alpha 32, constant LR 1.5e-5, 32 prompts with eight rollouts, 8192 response tokens, eight GPUs. Replace each abrupt 20-degree refresh with ten 2-degree fixed-plane increments after updates 40-49 and 80-89. Ordinary optimizer updates occur between increments. Retain B, compensate the frozen base to preserve effective weights in exact arithmetic, preserve Adam counters and A moments, approximately project B first moments by actual row cosine, and retain B variances. This is approximate moment transport; unseen gradient history is not reconstructed. Two compressed correction directions per cycle give rank capacity at most five including the active adapter, versus at most three for the one-shot trial. Native eight-rank FSDP validation passed, including exact recovery from an in-transition checkpoint and base-weight extraction; it did not launch an inference engine. AIME25 avg@8/pass@8 every twenty updates, hourly health checks, ten-second NVML sampling, and per-update training allocator peaks. Compare with existing historical standard LoRA, completed fresh standard, and one-shot refresh; no extra baseline rerun. Completed all 100 updates with successful exit. Final AIME25 avg@8/pass@8 is 15.42%/26.67%, below historical standard LoRA at 20.00%/43.33%. Mean accumulated stable rank is 1.02291 with only 2.2374% energy outside the leading direction. Training trends remain comparable, but the trial does not meet performance parity or demonstrate a substantial rank advantage. Logs, raw evaluations and audits are retained; completed large weights were removed.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-gradual-refresh-r1-20261004-01` |
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
| 20 | 5.8% | 20.0% |
| 40 | 13.8% | 33.3% |
| 60 | 16.2% | 33.3% |
| 80 | 16.2% | 36.7% |
| 100 | 15.4% | 26.7% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 4/240 | 3/30 |
| aime25 | 20 | 14/240 | 6/30 |
| aime25 | 40 | 33/240 | 10/30 |
| aime25 | 60 | 39/240 | 10/30 |
| aime25 | 80 | 39/240 | 11/30 |
| aime25 | 100 | 37/240 | 8/30 |

## Quantitative observations

Training response correctness averaged **10.2%** over the first 10 logged updates and **36.2%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.845 at step 1 → 0.1248 at step 100.
- Policy gradient norm: 0.03881 at step 1 → 0.0301 at step 100.
- Mean generated response tokens: 1218 at step 1 → 4376 at step 100.

Best recorded AIME25 pass@8: **36.7% at step 80**. Last recorded: **26.7% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

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
| 1-20 | 12.25% | 12.44% | +0.20 pp |
| 21-40 | 23.65% | 22.93% | -0.72 pp |
| 41-50 | 34.77% | 34.06% | -0.70 pp |
| 51-60 | 37.93% | 37.50% | -0.43 pp |
| 61-80 | 39.26% | 38.75% | -0.51 pp |
| 81-90 | 40.04% | 40.04% | +0.00 pp |
| 91-100 | 34.92% | 36.25% | +1.33 pp |

Latest common AIME25 evaluation: **step 100**.

| Metric | Gradual minus reference | Question-bootstrap 95% interval |
|---|---:|---:|
| avg@8 | -4.58 pp | [-8.33, -1.25] pp |
| avg@8: gain from step 0 | -3.75 pp | [-8.75, +0.83] pp |
| pass@8 | -16.67 pp | [-30.00, -3.33] pp |
| pass@8: gain from step 0 | -13.33 pp | [-36.67, +6.67] pp |

[Download all metric windows, question-level intervals, and configuration differences](../data/gradual-refresh-vs-historical-standard-analysis.json).

### Completed fresh standard LoRA

| Training updates | Reference correctness | Gradual correctness | Difference |
|---|---:|---:|---:|
| 1-20 | 12.21% | 12.44% | +0.23 pp |
| 21-40 | 21.37% | 22.93% | +1.56 pp |
| 41-50 | 32.54% | 34.06% | +1.52 pp |
| 51-60 | 36.33% | 37.50% | +1.17 pp |
| 61-80 | 38.44% | 38.75% | +0.31 pp |
| 81-90 | 41.25% | 40.04% | -1.21 pp |
| 91-100 | 35.70% | 36.25% | +0.55 pp |

Latest common AIME25 evaluation: **step 100**.

| Metric | Gradual minus reference | Question-bootstrap 95% interval |
|---|---:|---:|
| avg@8 | -2.08 pp | [-6.67, +3.33] pp |
| avg@8: gain from step 0 | -0.83 pp | [-6.25, +5.42] pp |
| pass@8 | -10.00 pp | [-23.33, +3.33] pp |
| pass@8: gain from step 0 | -6.67 pp | [-26.67, +16.67] pp |

[Download all metric windows, question-level intervals, and configuration differences](../data/gradual-refresh-vs-fresh-standard-analysis.json).

### Completed one-shot refresh

| Training updates | Reference correctness | Gradual correctness | Difference |
|---|---:|---:|---:|
| 1-20 | 12.07% | 12.44% | +0.37 pp |
| 21-40 | 24.71% | 22.93% | -1.78 pp |
| 41-50 | 37.73% | 34.06% | -3.67 pp |
| 51-60 | 38.52% | 37.50% | -1.02 pp |
| 61-80 | 38.46% | 38.75% | +0.29 pp |
| 81-90 | 41.09% | 40.04% | -1.05 pp |
| 91-100 | 36.17% | 36.25% | +0.08 pp |

Latest common AIME25 evaluation: **step 100**.

| Metric | Gradual minus reference | Question-bootstrap 95% interval |
|---|---:|---:|
| avg@8 | -1.67 pp | [-5.00, +2.08] pp |
| avg@8: gain from step 0 | +0.00 pp | [-3.33, +3.75] pp |
| pass@8 | -10.00 pp | [-23.33, +0.00] pp |
| pass@8: gain from step 0 | -6.67 pp | [-20.00, +6.67] pp |

[Download all metric windows, question-level intervals, and configuration differences](../data/gradual-refresh-vs-one-shot-analysis.json).

Question bootstrap retains eight-response groups but does not establish equivalence or account for training-seed variance. Startup, training correctness, and intermediate evaluations do not prove the final objective.

## Completed-run protocol and training allocator

Successful exit and **100 updates** were audited against the matched recipe. Raw AIME25 groups agree with logged scores at every configured evaluation, and all **800 memory records across 8 ranks** match the update and refresh schedule. This verifies the recorded protocol, not performance parity.

| Measurement | GiB | Step | Rank |
|---|---:|---:|---:|
| Maximum training allocation | 18.006 | 81 | 2 |
| Maximum training reservation | 18.945 | 81 | 2 |

These are policy-process allocator peaks in the worker's declared training window. They are not total GPU memory or a single-device/phone estimate; inference, weight synchronization, and export are outside this measurement. [Download the completion audit and source hashes](../data/qwen3-4b-base-grpo-gradual-refresh-r1-20261004-01-completion-audit.json).
