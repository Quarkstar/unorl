---
title: "GRPO · standard rank-1 ReLoRA, constant-LR resets"
---

# GRPO · standard rank-1 ReLoRA, constant-LR resets

Completed all 100 steps with exit status 0. Matched standard rank-one Kaiming LoRA / alpha 32 GRPO settings and complete AdamW state clears at merges 40/80; only post-merge restart ramp differs from the paired trial (disabled here). Successive 20-step training-correctness means: 11.895%, 22.051%, 32.793%, 37.324%, 37.031%. Final AIME25 avg@8 / pass@8: 17.083% / 30.0%, versus 15.833% / 40.0% for the ramp and 20.0% / 43.333% for standard LoRA. Step-80 evaluation peaked at 18.333% / 50.0%; final results fluctuate on 30 questions. Boundary response-prefix KL was 0.0005605 / 0.0005378 at steps 40/80, from 1024 total probe tokens per boundary. Final per-layer mean accumulated stable rank was 2.160, with 53.25% mean update energy outside the leading direction. Peak training allocated/reserved memory across eight ranks was 18.60/19.47 GiB, including scheduled merge diagnostics. Neither pair shows an immediate post-merge reward collapse or a clear improvement over the standard LoRA reference. Checkpoint/model weights and factor cache removed after completion. Cleanup also erroneously removed raw evaluation response dumps under exports; aggregate metrics, evaluation.jsonl, logs, memory traces and rank diagnostics survive. The cleanup rule now targets policy/critic weight directories and preserves benchmark dumps; a regression test verifies this. Response-level regrading is unavailable for this run.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-relora-r1-warmup0-20261002-01` |
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

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-relora-r1-warmup0-20261002-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-relora-r1-warmup0-20261002-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.7% | 10.0% |
| 20 | 5.8% | 26.7% |
| 40 | 10.4% | 36.7% |
| 60 | 12.5% | 23.3% |
| 80 | 18.3% | 50.0% |
| 100 | 17.1% | 30.0% |


## Quantitative observations

Training response correctness averaged **10.0%** over the first 10 logged updates and **34.9%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8485 at step 1 → 0.1293 at step 100.
- Policy gradient norm: 0.04319 at step 1 → 0.03432 at step 100.
- Mean generated response tokens: 1295 at step 1 → 3883 at step 100.

Best recorded AIME25 pass@8: **50.0% at step 80**. Last recorded: **30.0% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
