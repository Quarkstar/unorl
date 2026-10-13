---
title: "PPO · full-finetune verl recipe (collapsed)"
---

# PPO · full-finetune verl recipe (collapsed)

Full-parameter port of verl's PPO recipe to Qwen3-4B-Base and the DAPO math data: GAE single rollout, no KL loss, actor lr 1e-6, critic lr 1e-5, batch 1024/mini 256, clip 0.2, epochs 15, prompt 1024, response 8192. Stopped after 7 updates: mean response length rose from 1.6k to 7.5k tokens, training correctness fell from 0.101 to 0.048, and policy entropy fell from 0.92 to 0.18. Together with the earlier LoRA run the same collapse appears across LoRA versus full fine-tuning, KL on versus off, actor lr 3e-5 versus 1e-6, one versus five samples per prompt, and batch 256 versus 1024, while our GRPO runs (eight-sample group-mean baseline) are stable at the same 8192 response budget. This isolates the learned critic/GAE advantage estimator: on this hard long-horizon task it yields a length-biased advantage where the group baseline does not. Unverified hypothesis for later: SkyRL's PPO implementation may be faulty, since the same recipe trains in verl's own framework. PPO is set aside for now.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-ppo-full-verl-20261013-01` |
| Record | 8 steps logged |
| Group | Controlled follow-up experiments |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | gae |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 0 |
| LoRA alpha | — |
| LoRA initialization | kaiming (default) |
| Learning rate | 1e-06 |
| Warmup steps | 0 |
| Restart warmup (updates) | not applied |
| Merge/reset interval (updates) | not applied |
| Prompts × responses | 1024 × 1 = 1024 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 137 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-ppo-full-verl-20261013-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-ppo-full-verl-20261013-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

This snapshot contains no AIME25 checkpoint evaluation.

## Quantitative observations

Training response correctness averaged **7.7%** over the first 8 logged updates and **7.7%** over the last 8. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.7968 at step 1 → 0.1801 at step 8.
- Policy gradient norm: 0.7102 at step 1 → 0.4236 at step 8.
- Mean generated response tokens: 1212 at step 1 → 7514 at step 8.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
