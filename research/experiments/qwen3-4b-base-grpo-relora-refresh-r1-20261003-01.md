---
title: "GRPO · compensated gradual refresh, rank 1 (100 steps from base)"
---

# GRPO · compensated gradual refresh, rank 1 (100 steps from base)

Completed all 100 updates from base with exit status 0. Refresh at 40/80 rotates A by 20 degrees, retains B, compensates frozen backbone weights and approximately transports native AdamW moments; rank one, alpha 32, constant LR 1.5e-5, eight rollouts, 32 prompts/update, response budget 8192. Final AIME25 avg@8 / pass@8 is 17.083% / 36.667% (41/240 correct responses, 11/30 solved questions), versus historical standard LoRA 20.0% / 43.333%. The final 81-100 training correctness is higher, 38.633% versus 37.480%, with similar entropy and longer responses. The first-refresh 46-60 cold-reset deficit is absent, but the existing pre-refresh lead confounds attribution. Final mean accumulated stable rank is only 1.024, with 2.315% mean energy outside the leading direction; total update L2 is 3.379. Peak training allocated/reserved memory across all eight ranks is 18.002/18.947 GiB, measured for all 100 updates. Question-bootstrap endpoint avg@8 difference is -2.917 points with descriptive interval [-7.50,+1.25]; this is not proof of equivalence or superiority. Goal remains unmet. Completed weights, optimizer checkpoints and factor cache removed (35,450,884,601 bytes); all twelve evaluation JSONL files hash-verified unchanged, logs/metrics/diagnostics preserved, shared standard step-100 checkpoint untouched.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-relora-refresh-r1-20261003-01` |
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

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-relora-refresh-r1-20261003-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-relora-refresh-r1-20261003-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 3.3% | 13.3% |
| 20 | 5.8% | 20.0% |
| 40 | 17.1% | 26.7% |
| 60 | 17.5% | 36.7% |
| 80 | 17.1% | 30.0% |
| 100 | 17.1% | 36.7% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 8/240 | 4/30 |
| aime25 | 20 | 14/240 | 6/30 |
| aime25 | 40 | 41/240 | 8/30 |
| aime25 | 60 | 42/240 | 11/30 |
| aime25 | 80 | 41/240 | 9/30 |
| aime25 | 100 | 41/240 | 11/30 |

## Quantitative observations

Training response correctness averaged **9.7%** over the first 10 logged updates and **36.2%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8753 at step 1 → 0.1227 at step 100.
- Policy gradient norm: 0.02517 at step 1 → 0.03505 at step 100.
- Mean generated response tokens: 1339 at step 1 → 4213 at step 100.

Best recorded AIME25 pass@8: **36.7% at step 60**. Last recorded: **36.7% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
