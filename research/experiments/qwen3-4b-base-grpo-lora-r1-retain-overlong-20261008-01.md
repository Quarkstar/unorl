---
title: Retain-overlong standard rank-1 LoRA GRPO
---

# Retain-overlong standard rank-1 LoRA GRPO

## Motivation

SkyRL's `generator.apply_overlong_filtering` implements DAPO overlong filtering:
for any trajectory whose stop reason is not `stop` (i.e., truncated at the
8,192-token cap), the entire loss mask is zeroed, so the trajectory contributes
no policy-gradient update. Its reward is not modified and still enters the GRPO
group mean, so centering no longer holds on the retained trajectories.

The [post-step-60 analysis](../notes/batchnorm-after60.md) found that training
reward flattens while response length grows and the truncation rate rises; the
GRPO reference's truncation rate rose from 17.1% to 26.7% over steps 60–100.
This trial isolates the filtering choice: does retaining truncated failures
(train on their negative reward) change the length, truncation and plateau
behaviour relative to the filtered standard-LoRA baselines?

## Controlled recipe

A single change from the standard rank-1 LoRA GRPO baseline:
`generator.apply_overlong_filtering=false`. Every other setting matches
`configs/qwen3-4b-base-grpo-lora-r1-blog.json` — Qwen3-4B-Base, all-linear
rank-1 LoRA (alpha 32, Kaiming), native AdamW constant LR 1.5e-5, no warmup,
GRPO without group-standard-deviation normalization or KL, token-mean loss, no
policy clipping, token-level TIS correction with clip 1e9, 32 prompts × eight
responses, 8,192 response budget, seed 42, 100 updates, eight GPUs. AIME25
avg@8/pass@8 is evaluated every 20 updates.

## Run identity

- Run: `qwen3-4b-base-grpo-lora-r1-retain-overlong-20261008-01`
- Config: `configs/qwen3-4b-base-grpo-lora-r1-retain-overlong.json`
- Config commit: `8470aa5`
- Launcher: `python scripts/train.py --mode grpo` (plain `unorl.train`; no ReLoRA wrapper)
- SkyRL pin: `0b286bacba2bb51dfe50186b6b5d6b1e0b5f5518`
- Hardware: eight A100-80GB

## Comparisons

- Standard filtered baselines: historical rank-1 LoRA GRPO **20.0% / 43.3%**
  (avg@8/pass@8, step 100) and the fresh standard control **17.5% / 36.7%**.
- Mechanism reference: the truncation and coverage analysis in
  [post-step-60](../notes/batchnorm-after60.md).

## Pre-registered expectations

1. **Truncation/length.** If overlong filtering contributes to length growth,
   retaining truncated negatives should reduce the truncation rate and response
   length relative to the filtered baselines.
2. **Plateau.** Retaining adds gradient from truncated failures; the post-60
   training-correctness plateau may rise or fall. Direction is not assumed.
3. **Accuracy.** No directional claim. A single run on 30 questions cannot rank
   small endpoint differences.
4. **Diagnostics.** Record response length, the truncation rate where
   reconstructible, and the gap between completed-response and overall accuracy.

## Results

**Results are pending.** This page records the design and run identity for later
audit. Metrics, curves and figures will be added from the snapshot when all 100
updates and their evaluations complete. No outcome is assumed here.

## Provenance

The runtime records the resolved config, source snapshot, training-data audit and
launcher metadata under `runs/qwen3-4b-base-grpo-lora-r1-retain-overlong-20261008-01/`.
Because the run was launched before the config was committed, its recorded
`unorl-commit.txt` predates the config commit; the exact config that ran is
preserved in that run's `source/` snapshot and in commit `8470aa5`.
