---
title: "Early REINFORCE · loss-scale issue"
---

# Early REINFORCE · loss-scale issue

The sequence-summed loss and stateless SGD differ from the corrected experiment. Large logged gradient norms make it unsuitable for drawing conclusions about the later token-mean AdamW method.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-reinforce-lora-r1-blog-20260923-03` |
| Record | 100 steps logged / 100 planned |
| Group | Optimizer and loss confounds |
| Optimizer | SGD |
| Model | models/Qwen3-4B-Base |
| Training data | data/train-benchmark-clean.parquet |
| Advantage estimator | signed_reinforce |
| Policy loss | signed_reinforce |
| Loss reduction | seq_mean_token_sum_norm |
| LoRA rank | 1 |
| LoRA alpha | 32 |
| LoRA initialization | kaiming (default) |
| Learning rate | 1.5e-05 |
| Warmup steps | 0 |
| Prompts × responses | 32 × 1 = 32 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-reinforce-lora-r1-blog-20260923-03.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-reinforce-lora-r1-blog-20260923-03.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.1% | 16.7% |
| 20 | 0.8% | 6.7% |
| 40 | 2.5% | 10.0% |
| 60 | 2.9% | 13.3% |
| 80 | 1.7% | 13.3% |
| 100 | 2.5% | 16.7% |


Some raw evaluation dumps were truncated or malformed; aggregated logged metrics above are retained, but per-question counts for these files are unavailable:

- `exports/aime25/dumped_evals/global_step_100_evals/aime25.jsonl`

## Evaluation sample counts

| Benchmark | Step | Correct responses | Questions solved ≥1 time |
|---|---:|---:|---:|
| aime25 | 0 | 5/240 | 5/30 |
| aime25 | 20 | 2/240 | 2/30 |
| aime25 | 40 | 6/240 | 3/30 |
| aime25 | 60 | 7/240 | 4/30 |
| aime25 | 80 | 4/240 | 4/30 |

## Quantitative observations

Training response correctness averaged **8.4%** over the first 10 logged updates and **7.8%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.675 at step 1 → 0.7453 at step 100.
- Policy gradient norm: 198.6 at step 1 → 221.9 at step 100.
- Mean generated response tokens: 739.8 at step 1 → 1016 at step 100.

Best recorded AIME25 pass@8: **16.7% at step 0**. Last recorded: **16.7% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
