---
title: "GRPO · rank-1 LoRA in the final 18 layers"
---

# GRPO · rank-1 LoRA in the final 18 layers

In-progress controlled comparison with full-layer rank-1 GRPO. The same model, data, optimizer, learning rate, rollout count and response budget are used; adapters are excluded from layers 0–17. The actual synced adapter contains 1,032,192 parameters in layers 18–35, versus 2,064,384 in the reference. Initial learning is slower, while training correctness and held-out accuracy improve. This partial snapshot does not establish the final performance gap or peak training memory savings. Failed startup attempts produced no learning results and are excluded.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-lora-r1-last18-20260928-01` |
| Record | 83 steps logged / 100 planned |
| Group | Controlled follow-up experiments |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | grpo |
| Policy loss | regular |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 32 |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | not retained |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-lora-r1-last18-20260928-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-lora-r1-last18-20260928-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.2% | 10.0% |
| 20 | 2.1% | 10.0% |
| 40 | 7.5% | 20.0% |
| 60 | 12.1% | 26.7% |
| 80 | 15.0% | 26.7% |


Some raw evaluation dumps were truncated or malformed; aggregated logged metrics above are retained, but per-question counts for these files are unavailable:

- `exports/aime25/dumped_evals/global_step_0_evals/aime25.jsonl`

## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 20 | 5/240 | 3/30 |
| aime25 | 40 | 18/240 | 6/30 |
| aime25 | 60 | 29/240 | 8/30 |
| aime25 | 80 | 36/240 | 8/30 |

## Quantitative observations

Training response correctness averaged **9.0%** over the first 10 logged updates and **34.6%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8409 at step 1 → 0.1352 at step 83.
- Policy gradient norm: 0.02289 at step 1 → 0.02938 at step 83.
- Mean generated response tokens: 1243 at step 1 → 3636 at step 83.

Best recorded AIME25 pass@8: **26.7% at step 60**. Last recorded: **26.7% at step 80**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
