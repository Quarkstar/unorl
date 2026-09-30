---
title: "PPO · value-model warmup"
---

# PPO · value-model warmup

The run was stopped after early degradation. Five critic-only warmup steps precede actor updates; critic and policy settings differ from REINFORCE. This is evidence about this PPO configuration, not a general failure of PPO.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-ppo-lora-r1-valuewarmup-20260924-01` |
| Record | 38 steps logged / 105 planned |
| Group | Earlier research |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | gae |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 32 |
| LoRA initialization | kaiming (default) |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 256 × 1 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | not retained |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-ppo-lora-r1-valuewarmup-20260924-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-ppo-lora-r1-valuewarmup-20260924-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 3.3% | 16.7% |
| 25 | 1.2% | 10.0% |


Some raw evaluation dumps were truncated or malformed; aggregated logged metrics above are retained, but per-question counts for these files are unavailable:

- `exports/aime25/dumped_evals/global_step_25_evals/aime25.jsonl`

## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 8/240 | 5/30 |

## Quantitative observations

Training response correctness averaged **8.7%** over the first 10 logged updates and **0.9%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8498 at step 6 → 3.208 at step 38.
- Policy gradient norm: 0.203 at step 6 → 0.06331 at step 38.
- Mean generated response tokens: 1355 at step 1 → 8151 at step 38.

Best recorded AIME25 pass@8: **16.7% at step 0**. Last recorded: **10.0% at step 25**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
