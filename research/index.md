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

## Full-parameter and LoRA comparisons

All rows use Qwen3-4B-Base and eight rollouts per prompt. Full-parameter GRPO is a historical reference: LR, warmup, advantage normalization, clipping and importance correction differ from the later LoRA recipe. The LoRA-FA run completes at 17.5% avg@8 and 33.3% pass@8; its step-80 advantage does not persist at step 100. Single runs on 30 questions cannot establish a reliable ranking of small differences.

| Method | Step-80 avg@8 | Step-100 avg@8 | Step-100 pass@8 |
|---|---:|---:|---:|
| [Full-parameter GRPO](experiments/qwen3-4b-base-grpo-20260916-01.md) | 16.7% | 17.9% | 36.7% |
| [Rank-1 LoRA GRPO](experiments/qwen3-4b-base-grpo-lora-r1-blog-20260923-01.md) | 13.8% | 20.0% | 43.3% |
| [GRPO · rank-1 NoRA-init](experiments/qwen3-4b-base-grpo-lora-r1-nora-init-20260928-01.md) | 15.4% | 17.9% | 36.7% |
| [GRPO · full-layer rank-1 LoRA-FA](experiments/qwen3-4b-base-grpo-lorafa-r1-20260930-01.md) | 17.9% | 17.5% | 33.3% |
| [GRPO · rank-1 LoRA in the final 18 layers](experiments/qwen3-4b-base-grpo-lora-r1-last18-20260928-01.md) | 15.0% | 17.9% | 33.3% |
| [GRPO · rank-1 NoRA-init in the final 18 layers](experiments/qwen3-4b-base-grpo-lora-r1-nora-init-last18-20260929-01.md) | 16.2% | 15.8% | 30.0% |

```{figure} figures/comparison-lora.svg
:alt: Google Material palette. Full-parameter GRPO remains visible as a historical reference with different settings.

Google Material palette. Full-parameter GRPO remains visible as a historical reference with different settings.
```

## NoRA-init: faster early learning

The completed [NoRA-init trial](experiments/qwen3-4b-base-grpo-lora-r1-nora-init-20260928-01.md) reaches **33.8%** mean training correctness in steps 21–40, versus **23.7%** with standard rank-1 LoRA. Step-20 AIME25 avg@8 is **9.2% versus 4.2%**. Later training correctness plateaus near 38–39%; final avg@8 / pass@8 is **17.9% / 36.7%**, versus **20.0% / 43.3%**. Initialization and alpha differ together. The early acceleration motivates testing merge/reset; the cause of the plateau and benefit of merging remain unproven.

```{figure} figures/comparison-nora.svg
:alt: Full-layer and final-half NoRA-init, compared with standard full-layer rank-1 LoRA GRPO.

Full-layer and final-half NoRA-init, compared with standard full-layer rank-1 LoRA GRPO.
```

## All retained experiment results

Final columns use the last recorded AIME25 evaluation, whose step is shown separately from the last training step. A dash means missing evidence, not zero accuracy. Smoke tests are excluded.

### Controlled follow-up experiments

| Experiment | Last train step | Last eval step | Avg@8 | Pass@8 | Optimizer |
|---|---:|---:|---:|---:|---|
| [GRPO · full-layer rank-1 LoRA-FA](experiments/qwen3-4b-base-grpo-lorafa-r1-20260930-01.md) | 100 | 100 | 17.5% | 33.3% | AdamW (SkyRL default) |
| [GRPO · rank-1 NoRA-init in the final 18 layers](experiments/qwen3-4b-base-grpo-lora-r1-nora-init-last18-20260929-01.md) | 100 | 100 | 15.8% | 30.0% | AdamW (SkyRL default) |
| [GRPO · rank-1 NoRA-init](experiments/qwen3-4b-base-grpo-lora-r1-nora-init-20260928-01.md) | 100 | 100 | 17.9% | 36.7% | AdamW (SkyRL default) |
| [GRPO · rank-1 LoRA in the final 18 layers](experiments/qwen3-4b-base-grpo-lora-r1-last18-20260928-01.md) | 100 | 100 | 17.9% | 33.3% | AdamW (SkyRL default) |
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

## Research directions and next questions

The [2025–2026 LoRA training investigation](notes/lora-training-2025-2026.md) compares LoRA-FA, LoFT, recent optimizer-state research, and merge/reset designs. It separates published evidence from proposed UNORL experiments.

1. Final-layer LoRA: the last-half trial completed; measure actual activation/peak memory savings and investigate fewer layers.
2. [NoRA initialization](notes/nora.md): the trial completed with promising early acceleration. Next, test periodic merge/reset as a possible way to sustain learning; this is a hypothesis, not an executed follow-up.
3. Test QLoRA separately; this direction remains untested.

The current scope is on-policy learning; small batches and single-rollout use remain central.

[Measurement conventions](methods.md) · [Build and publish](publishing.md) · [Download comparison data](data/comparison.csv)
