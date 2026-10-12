---
title: "PPO · worked rank-32 LoRA recipe (collapsed)"
---

# PPO · worked rank-32 LoRA recipe (collapsed)

Port of SkyRL's worked LoRA-PPO example to Qwen3-4B-Base and the DAPO math data: rank-32 LoRA policy and critic, use_kl_loss true, five samples per prompt, GAE (gamma 1.0, lambda 0.95), clip 0.2, policy and critic lr 3e-5, epochs 1, prompt 2048, response 8192, batch 256, AIME25+AIME26 (60 questions) eval. Stopped after 12 of 29 updates once the trajectory collapsed: mean response length exploded from 1.3k to 8.0k tokens, training correctness fell from 0.065 to 0.017, policy entropy rose 0.86 to 1.24 then collapsed to 0.19, and AIME25/AIME26 avg@8 fell to 0.42%/0.0%. The KL term was inactive (kl_loss_coef default 0.001, penalty about 1e-6) and the measured policy KL stayed small (at most 0.0068, then fell), so the collapse is length/objective driven, not drift away from the reference. Negative result: the worked recipe transfers to easy short GSM8K with a small Instruct model but does not transfer to hard long-response RL on a 4B base model. The 3e-5 LR is standard for LoRA and is not the cause.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-ppo-lora-r32-example-20261012-01` |
| Record | 12 steps logged |
| Group | Controlled follow-up experiments |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | gae |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| LoRA initialization | kaiming (default) |
| Learning rate | 3e-05 |
| Warmup steps | 0 |
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 256 × 5 = 1280 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | True / False |
| Exit status | 137 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-ppo-lora-r32-example-20261012-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-ppo-lora-r32-example-20261012-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 5 | 0.8% | 6.7% |
| 10 | 0.4% | 3.3% |

### AIME26

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 5 | 2.5% | 10.0% |
| 10 | 0.0% | 0.0% |


## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 5 | 2/240 | 2/30 |
| aime25 | 10 | 1/240 | 1/30 |
| aime26 | 5 | 6/240 | 3/30 |
| aime26 | 10 | 0/240 | 0/30 |

## Quantitative observations

Training response correctness averaged **5.5%** over the first 10 logged updates and **4.3%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8766 at step 1 → 0.1879 at step 12.
- Policy gradient norm: 0.0445 at step 1 → 0.06145 at step 12.
- Mean generated response tokens: 1201 at step 1 → 8037 at step 12.

Best recorded AIME25 pass@8: **6.7% at step 5**. Last recorded: **3.3% at step 10**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
