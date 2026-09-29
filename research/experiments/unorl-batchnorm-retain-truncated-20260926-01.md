---
title: "Retain truncated failures · AdamW"
---

# Retain truncated failures · AdamW

Completed 100-step controlled follow-up. Only overlong filtering changes from the batch-normalized AdamW profile. Training masks confirm truncated failures receive updates. AIME25 truncation fell substantially by step 100, but final avg@8 was 14.6% versus 17.9% with filtering; both had 30.0% pass@8. Steps 60–100 were steadier in pass@8, but accuracy did not show stable improvement. See the batch-normalized REINFORCE diagnosis.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `unorl-batchnorm-retain-truncated-20260926-01` |
| Record | 100 steps logged / 100 planned |
| Group | Controlled follow-up experiments |
| Optimizer | AdamW (SkyRL default) |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | batch_norm_reinforce |
| Policy loss | signed_reinforce |
| Loss reduction | token_mean |
| LoRA rank | 1 |
| LoRA alpha | 32 |
| LoRA initialization | kaiming (default) |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Prompts × responses | 256 × 1 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/unorl-batchnorm-retain-truncated-20260926-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/unorl-batchnorm-retain-truncated-20260926-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 1.7% | 10.0% |
| 20 | 8.3% | 23.3% |
| 40 | 11.7% | 30.0% |
| 60 | 15.8% | 23.3% |
| 80 | 16.2% | 30.0% |
| 100 | 14.6% | 30.0% |


Some raw evaluation dumps were truncated or malformed; aggregated logged metrics above are retained, but per-question counts for these files are unavailable:

- `exports/aime25/dumped_evals/global_step_0_evals/aime25.jsonl`
- `exports/aime25/dumped_evals/global_step_20_evals/aime25.jsonl`

## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 40 | 28/240 | 9/30 |
| aime25 | 60 | 38/240 | 7/30 |
| aime25 | 80 | 39/240 | 9/30 |
| aime25 | 100 | 35/240 | 9/30 |

## Quantitative observations

Training response correctness averaged **9.7%** over the first 10 logged updates and **37.7%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.8443 at step 1 → 0.1325 at step 100.
- Policy gradient norm: 0.06362 at step 1 → 0.04122 at step 100.
- Mean generated response tokens: 1077 at step 1 → 3487 at step 100.

Best recorded AIME25 pass@8: **30.0% at step 40**. Last recorded: **30.0% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
