---
title: "GRPO · standard rank-1 ReLoRA, five-update restart ramp"
---

# GRPO · standard rank-1 ReLoRA, five-update restart ramp

Completed all 100 steps with exit status 0 after fixing the trajectory-prefix diagnostic. Standard Kaiming rank-one LoRA / alpha 32, native AdamW LR 1.5e-5, eight rollouts, 32 prompts/update, 8192 response tokens; merge/reset at 40/80, then five-update LR multipliers 0, .25, .5, .75, 1; no initial warmup. Successive 20-step training-correctness means: 12.656%, 23.281%, 32.695%, 36.348%, 36.309%. Final AIME25 avg@8 / pass@8: 15.833% / 40.0%, versus 17.083% / 30.0% for constant-LR resets and 20.0% / 43.333% for standard LoRA. There is no clear benefit from this ramp in one sampled run per setting. Boundary response-prefix KL was 0.0005664 / 0.0005645 at steps 40/80, using 1024 real response-prefix tokens per boundary. Final mean accumulated stable rank was 1.887, with 46.48% mean update energy outside the leading direction. Rank growth is measurable but did not establish better learning. Peak training allocated/reserved memory across eight ranks was 17.96/18.95 GiB. Neither merge produced an immediate reward collapse. Earlier failed attempt is separately retained and never reached a merge. Checkpoint/model weights and factor cache removed after completion. Cleanup also erroneously removed raw evaluation response dumps under exports; aggregate metrics, evaluation.jsonl, logs, memory traces and rank diagnostics survive. Cleanup was fixed and regression-tested. Response-level regrading is unavailable for this run.

## Configuration and provenance

| Setting | Value |
|---|---|
| Run ID | `qwen3-4b-base-grpo-relora-r1-warmup5-20261002-01` |
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
| Restart warmup (updates) | 5 |
| Merge/reset interval (updates) | 40 |
| Prompts × responses | 32 × 8 = 256 responses/update |
| Response limit | 8192 |
| Evaluation samples/question | 8 |
| GPUs (policy) | 8 |
| KL loss / reward | False / False |
| Exit status | 0 |

[Download the metric/configuration snapshot](../data/qwen3-4b-base-grpo-relora-r1-warmup5-20261002-01.json). The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.

## Learning curves

```{figure} ../figures/qwen3-4b-base-grpo-relora-r1-warmup5-20261002-01.svg
:alt: Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.

Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.
```

## Evaluation results

### AIME25

| Step | Sample accuracy | Pass@8 |
|---:|---:|---:|
| 0 | 2.5% | 10.0% |
| 20 | 7.5% | 26.7% |
| 40 | 12.9% | 33.3% |
| 60 | 13.8% | 33.3% |
| 80 | 16.2% | 30.0% |
| 100 | 15.8% | 40.0% |


## Quantitative observations

Training response correctness averaged **9.5%** over the first 10 logged updates and **33.1%** over the last 10. These are different on-policy training batches, so this trend is not a fixed-test comparison.

- Logged entropy: 0.807 at step 1 → 0.1288 at step 100.
- Policy gradient norm: 0.02314 at step 1 → 0.05443 at step 100.
- Mean generated response tokens: 1448 at step 1 → 3968 at step 100.

Best recorded AIME25 pass@8: **40.0% at step 100**. Last recorded: **40.0% at step 100**. Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.

## Interpretation limits

One retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).
