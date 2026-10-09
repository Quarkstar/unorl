---
title: The research program
---

# On-device continual reinforcement learning

## The goal

Train and keep adapting a language model with reinforcement learning under
**extreme resource limits — ultimately on a phone** — so that it can learn
**continually** where it runs, without a datacenter. The target is not a
benchmark score; it is a *feasible* training loop: bounded rollout memory,
bounded optimizer memory, bounded activation memory, and a process that keeps
improving instead of saturating.

## Where the memory goes

1. **Rollouts and activations** — generating and back-propagating many long
   responses per prompt.
2. **Optimizer state** — Adam moments and counters scale with trainable
   parameters.
3. **Parameters and gradients.**

## Two levers on the budget

### Lever 1 — single-rollout RL (algorithmic)

Use **one response per prompt** instead of a group of eight, cutting rollout
memory and batch size. This raises gradient variance, so the objective has to be
variance-reduced and is probably **PPO-like** (clipped surrogate, a baseline, and
possibly a small critic). The retained single-rollout evidence so far uses
REINFORCE-style objectives; a PPO-like single-rollout design is the leading open
hypothesis, not yet a demonstrated method.

### Lever 2 — LoRA / ReLoRA (parametric)

Optimize a **rank-1 adapter** so optimizer state is tiny, and periodically
**merge and reset** (ReLoRA) to accumulate effective rank beyond one. The risk is
a capability ceiling: rank one may not hold what the task needs.

## What we found

- **Rank-1 LoRA matches full-parameter GRPO** at 100 updates (single runs on 30
  questions; not a reliable ranking of small differences).
- **Single-rollout batch-normalized REINFORCE with AdamW is competitive** with
  eight-rollout GRPO. Earlier SGD trials are separated because they do not
  isolate the algorithm.
- **Saturation:** training reward plateaus around **steps 50–60** for LoRA,
  full-parameter, and REINFORCE alike; a 100→200 continuation is flat.
- **ReLoRA/merge** grows the accumulated stable rank to about two but does **not**
  improve the endpoint; compensated refresh keeps parity.
- **Length/truncation is a small measurement effect**: doubling the generation
  budget removes truncation but moves accuracy only one to two points.

## The central open problem

If full-parameter and rank-1 LoRA both saturate, then **rank is not the
limiter**. Under this setting the limiting factor is likely the **data, the
objective, or the horizon** — not the parameterization. Reaching capability under
a phone-scale budget may require a **new setting** (harder or less-saturated
data, a longer meaningful horizon, or an objective that keeps the reward signal
alive), rather than another adapter variant.

## Success criteria

1. **Feasibility** — the loop fits the memory budget; report measured peaks, not
   estimates.
2. **Continual learning** — reward does **not** saturate; it keeps improving over
   a long horizon.
3. **Parity** — the low-resource method matches a strong reference at matched
   compute.

## Open directions

1. **Single-rollout PPO-like algorithm** — the under-explored algorithmic lever.
2. **A setting that rewards capacity** — so that LoRA vs full-parameter can
   differ at all.
3. **Explicit on-device memory accounting** — turn the resource claim into a
   measurement.
4. **Optimizer-state transport across ReLoRA resets** — only if capacity is shown
   to matter.
