---
title: "GRPO · ReLoRA restart ramp, failed probe attempt"
---

# GRPO · ReLoRA restart ramp, failed probe attempt

Stopped with exit status 1 during step 40, before the first merge or optimizer update at that step; 39 updates completed. A diagnostic indexing bug treated the start of the common response slice as the prompt boundary. With native SkyRL whole-sequence left padding, short responses can have their prompt inside that slice, so the probe incorrectly raised an empty-prompt error. Native training itself was unaffected. Correctness over steps 1–10, 11–20, 21–30 and 31–39 was 10.4%, 17.1%, 21.1%, 27.7%; standard LoRA reference over the same ranges was 9.65%, 14.84%, 20.82%, 25.95%. AIME25 at step 20 was 6.25% avg@8 / 26.67% pass@8, versus 4.17% / 23.33% in the reference. These are pre-merge single-run results and do not establish a ReLoRA benefit. No boundary/rank results were produced. The control did not start. Fixed and restarted from the base model on October 2; original logs and evaluations retained.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-relora-r1-warmup5-20261001-01` |
| Record | 39 steps logged / 100 planned |
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
| Exit status | 1 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-relora-r1-warmup5-20261001-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-relora-r1-warmup5-20261001-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.5% | 16.7% |
| 20 | 6.2% | 26.7% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 6/240 | 5/30 |
| aime25 | 20 | 15/240 | 8/30 |

## Quantitative observations

Training response correctness averaged **10.4%** over the first 10 logged updates and **27.3%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8396 at step 1 → 0.2061 at step 39.
- Policy gradient norm: 0.04044 at step 1 → 0.02973 at step 39.
- Mean generated response tokens: 1354 at step 1 → 2821 at step 39.

Best recorded AIME25 pass@8: **26.7% at step 20**. Last recorded: **26.7% at step 20**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
