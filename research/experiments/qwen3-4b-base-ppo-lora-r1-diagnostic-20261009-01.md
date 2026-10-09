---
title: "PPO · rank-1 critic diagnostic (single rollout)"
---

# PPO · rank-1 critic diagnostic (single rollout)

Twenty-update diagnostic of the single-rollout PPO line. Qwen3-4B-Base, rank-one LoRA policy and a separate rank-one LoRA critic, GAE (gamma 1.0, lambda 0.95), clip 0.2, advantage batch-normalize, one rollout per prompt, five frozen critic-warmup steps. Added read-only diagnostics logged the critic value against the terminal reward. The critic stayed near zero (value mean -0.002 to -0.09, std 0.01 to 0.09) and only weakly correlated with reward (0.05-0.16) over the run, so GAE advantages were poor; policy entropy rose 0.85 to 1.99, training correctness fell 0.09 to 0.027, and AIME25 avg@8 fell from 2.08% to 0.42% with degenerate generations. Evidence that a rank-one critic cannot support GAE in this regime, not a general failure of PPO; the critic-free batch-normalized REINFORCE baseline is stronger. AIME25+AIME26 (60 questions) avg@8.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-ppo-lora-r1-diagnostic-20261009-01` |
| Record | 20 steps logged / 20 planned |
| Group | Controlled follow-up experiments |
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
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 256 × 1 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-ppo-lora-r1-diagnostic-20261009-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-ppo-lora-r1-diagnostic-20261009-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.1% | 10.0% |
| 10 | 2.1% | 13.3% |
| 20 | 0.4% | 3.3% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.1% | 13.3% |
| 10 | 1.2% | 10.0% |
| 20 | 2.5% | 16.7% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 5/240 | 3/30 |
| aime25 | 10 | 5/240 | 4/30 |
| aime25 | 20 | 1/240 | 1/30 |
| aime26 | 0 | 5/240 | 4/30 |
| aime26 | 10 | 3/240 | 3/30 |
| aime26 | 20 | 6/240 | 5/30 |

## Quantitative observations

Training response correctness averaged **8.5%** over the first 10 logged updates and **5.4%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.9551 at step 6 → 2.277 at step 20.
- Policy gradient norm: 0.1924 at step 6 → 0.09358 at step 20.
- Mean generated response tokens: 1231 at step 1 → 5522 at step 20.

Best recorded AIME25 pass@8: **13.3% at step 10**. Last recorded: **3.3% at step 20**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
