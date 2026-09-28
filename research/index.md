---
title: UNORL research book
---

# UNORL

**Resource-efficient, on-policy reinforcement learning.** This book connects each experiment to its configuration, measured learning curves, and evaluation results. It builds from small versioned data snapshots; weights and generated solution text stay outside the repository.

The present evidence supports learning with rank-1 LoRA and single-rollout REINFORCE using AdamW. At step 100, batch-normalized REINFORCE reaches AIME25 avg@8 **17.9%** and pass@8 **30.0%**, versus vanilla's **14.6% / 26.7%** and rank-1 GRPO's **20.0% / 43.3%**. These are single-run observations on 30 questions, not a reliable ranking of small differences. Earlier SGD trials are explicitly separated because they do not isolate algorithm choice.

## Main comparison

GRPO uses 32 prompts × 8 responses; REINFORCE uses 256 prompts × 1 response. Both generate 256 responses per update. The full-parameter reference uses different optimizer settings; all records expose their settings below.

```{figure} figures/comparison-primary.svg
:alt: Google Material palette. Evaluation points are unsmoothed; training curves use a trailing 10-update mean with raw values faintly shown.

Google Material palette. Evaluation points are unsmoothed; training curves use a trailing 10-update mean with raw values faintly shown.
```

## All retained experiment results

Final columns use the last recorded AIME25 evaluation, whose step is shown separately from the last training step. A dash means missing evidence, not zero accuracy. Smoke tests are excluded.

### Controlled follow-up experiments

| Experiment | Last train step | Last eval step | Avg@8 | Pass@8 | Optimizer |
|---|---:|---:|---:|---:|---|
| [GRPO · rank-1 LoRA in the final 18 layers](experiments/qwen3-4b-base-grpo-lora-r1-last18-20260928-01.md) | 83 | 80 | 15.0% | 26.7% | AdamW (SkyRL default) |
| [Retain truncated failures · AdamW](experiments/unorl-batchnorm-retain-truncated-20260926-01.md) | 100 | 100 | 14.6% | 30.0% | AdamW (SkyRL default) |

### Current algorithm comparison

| Experiment | Last train step | Last eval step | Avg@8 | Pass@8 | Optimizer |
|---|---:|---:|---:|---:|---|
| [Rank-1 LoRA GRPO](experiments/qwen3-4b-base-grpo-lora-r1-blog-20260923-01.md) | 100 | 100 | 20.0% | 43.3% | AdamW (SkyRL default) |
| [Vanilla REINFORCE · AdamW](experiments/qwen3-4b-base-reinforce-adamw-r1-b256-20260925-01.md) | 100 | 100 | 14.6% | 26.7% | AdamW (SkyRL default) |
| [Batch-normalized REINFORCE · AdamW](experiments/qwen3-4b-base-reinforce-batchnorm-adamw-r1-b256-20260925-01.md) | 100 | 100 | 17.9% | 30.0% | AdamW (SkyRL default) |

### Full-parameter reference

| Experiment | Last train step | Last eval step | Avg@8 | Pass@8 | Optimizer |
|---|---:|---:|---:|---:|---|
| [Full-parameter GRPO](experiments/qwen3-4b-base-grpo-20260916-01.md) | 100 | 100 | 17.9% | 36.7% | AdamW (SkyRL default) |

### Earlier research

| Experiment | Last train step | Last eval step | Avg@8 | Pass@8 | Optimizer |
|---|---:|---:|---:|---:|---|
| [Archived · conditional online SFT](experiments/qwen3-4b-base-conditional-20260916-01.md) | 300 | 300 | 0.0% | 0.0% | AdamW (SkyRL default) |
| [Archived · positive-only online SFT](experiments/qwen3-4b-base-positive-only-20260918-01.md) | 300 | 300 | 15.4% | 30.0% | AdamW (SkyRL default) |
| [PPO · value-model warmup](experiments/qwen3-4b-base-ppo-lora-r1-valuewarmup-20260924-01.md) | 38 | 25 | 1.2% | 10.0% | AdamW (SkyRL default) |
| [Archived · Instruct GRPO pilot](experiments/qwen3-4b-instruct-grpo-20260915-01.md) | 12 | 0 | 45.0% | 66.7% | AdamW (SkyRL default) |

### Optimizer and loss confounds

| Experiment | Last train step | Last eval step | Avg@8 | Pass@8 | Optimizer |
|---|---:|---:|---:|---:|---|
| [Batch-normalized REINFORCE · SGD](experiments/qwen3-4b-base-reinforce-batchnorm-r1-b256-20260925-02.md) | 100 | 100 | 2.1% | 6.7% | SGD |
| [Mean-centered REINFORCE · SGD](experiments/qwen3-4b-base-reinforce-batchmean-b256-20260924-01.md) | 22 | 20 | 1.7% | 13.3% | SGD |
| [Token-mean REINFORCE · SGD](experiments/qwen3-4b-base-reinforce-tokenmean-b256-20260924-01.md) | 73 | 60 | 2.9% | 13.3% | SGD |
| [Early REINFORCE · loss-scale issue](experiments/qwen3-4b-base-reinforce-lora-r1-blog-20260923-03.md) | 100 | 100 | 2.5% | 16.7% | SGD |
| [Single-GPU ReLoRA exploration](experiments/qwen3-4b-base-reinforce-relora-gpu0-b16-m100-300-20260922-01.md) | 295 | 200 | 1.7% | 6.7% | SGD |

### Incomplete launch records

| Experiment | Last train step | Last eval step | Avg@8 | Pass@8 | Optimizer |
|---|---:|---:|---:|---:|---|
| [Single-GPU LoRA launch record](experiments/qwen3-4b-base-grpo-lora-r1-gpu0-20260923-01.md) | — | — | — | — | AdamW (SkyRL default) |
| [Interrupted batch-normalized launch](experiments/qwen3-4b-base-reinforce-batchnorm-r1-b256-20260925-01.md) | — | — | — | — | SGD |

```{figure} figures/comparison-history.svg
:alt: Earlier research used different objectives, model variants, and response budgets. These curves provide context, not a controlled ranking.

Earlier research used different objectives, model variants, and response budgets. These curves provide context, not a controlled ranking.
```
```{figure} figures/comparison-confounded.svg
:alt: Earlier single-rollout trials used stateless SGD and sometimes a different loss reduction. Their failures do not establish that REINFORCE fails with AdamW.

Earlier single-rollout trials used stateless SGD and sometimes a different loss reduction. Their failures do not establish that REINFORCE fails with AdamW.
```

## Post-step-60 diagnosis

The [truncation analysis](notes/batchnorm-after60.md) examines the loss of question coverage and specifies a controlled follow-up trial.

## Next research questions

1. Measure adapter magnitudes by layer and test LoRA on only the final N layers.
2. Test [NoRA-style initialization](notes/nora.md) with scaling controlled.
3. Test QLoRA separately.

The current scope is on-policy learning; small batches and single-rollout use remain central. These next experiments are planned, not executed results.

[Measurement conventions](methods.md) · [Build and publish](publishing.md) · [Download comparison data](data/comparison.csv)
