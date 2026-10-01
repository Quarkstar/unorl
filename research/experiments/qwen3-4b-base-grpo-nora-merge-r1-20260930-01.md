---
title: "GRPO · full NoRA-init with merge/reset"
---

# GRPO · full NoRA-init with merge/reset

Completed all 100 steps successfully in 6h43m of training-loop time. Merges at steps 40 and 80 accumulated each learned adapter, created fresh normalized sign A / zero B, and cleared adapter AdamW history while preserving constant LR/scheduler. Backbone synchronization succeeded at startup and both merges. Training correctness over successive 20-step windows was 18.4%, 34.4%, 39.2%, 39.6%, and 37.2%: this run did not sustain renewed improvement after resets. AIME25 avg@8 peaked at 20.0% at step 80, then finished at 15.0% / 30.0% avg@8 / pass@8, versus 17.9% / 36.7% without merges and 20.0% / 43.3% for standard LoRA. These are sampled single-run results on 30 questions. Merge probe max logit differences were 0.25 and 0.3125, with RMS differences 0.0506 and 0.0556; merging was not bitwise function-preserving under mixed-precision forward computation. Peak training allocator usage was 18.00 GiB allocated / 18.96 GiB reserved, including merge work. This explicit Adam-reset baseline does not isolate rank growth from optimizer discontinuity or establish that merging generally fails.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-nora-merge-r1-20260930-01` |
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
| Merge/reset interval (updates) | 40 |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-nora-merge-r1-20260930-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-nora-merge-r1-20260930-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.1% | 13.3% |
| 20 | 16.2% | 33.3% |
| 40 | 16.2% | 33.3% |
| 60 | 15.4% | 30.0% |
| 80 | 20.0% | 36.7% |
| 100 | 15.0% | 30.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 5/240 | 4/30 |
| aime25 | 20 | 39/240 | 10/30 |
| aime25 | 40 | 39/240 | 10/30 |
| aime25 | 60 | 37/240 | 9/30 |
| aime25 | 80 | 48/240 | 11/30 |
| aime25 | 100 | 36/240 | 9/30 |

## Quantitative observations

Training response correctness averaged **11.7%** over the first 10 logged updates and **33.4%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8427 at step 1 → 0.1097 at step 100.
- Policy gradient norm: 0.1072 at step 1 → 0.1028 at step 100.
- Mean generated response tokens: 1285 at step 1 → 5282 at step 100.

Best recorded AIME25 pass@8: **36.7% at step 80**. Last recorded: **30.0% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
