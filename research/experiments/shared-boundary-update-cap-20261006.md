---
title: Shared step-40 boundary · ReLoRA update-size control
---

# Shared step-40 boundary · ReLoRA update-size control

## Motivation

The [matched shared-boundary comparison](shared-boundary-relora-20261006.md)
shows that ordinary random-reset ReLoRA can accumulate meaningful rank while
retaining similar endpoint AIME25 accuracy in one run. However, its first
post-reset effective weight update is much larger than standard LoRA's.

| Update | Standard LoRA weight-update L2 | Ordinary ReLoRA weight-update L2 |
|---:|---:|---:|
| 41 | 0.06674 | 0.28854 |
| 81 | 0.06081 | 0.28872 |

This is a measured continuity mismatch, not proof that update size causes a
reward deficit. The next experiment isolates a weight-update cap while keeping
the ordinary random-reset directions and native AdamW state rules.

## Controlled recipe

Resume the same unmerged native checkpoint at global step 40 and stop at
global step 100. Use all eight GPUs, rank-one Kaiming LoRA with alpha 32,
eight rollouts, 32 prompts per batch, 8,192 response tokens, native GRPO,
no KL, constant AdamW LR 1.5e-5, no warmup, and the existing complete-batch
calibration at boundaries 40/80. Merge and reset before updates 41/81,
clearing Adam exactly as in ordinary ReLoRA.

The configuration differs from the completed ordinary-reset reference only
by `trainer.update_cap_budget_ratio=1.0`, using the update-cap worker.
No new standard or ordinary control run is queued. Their completed results
remain the references; sampling variation limits any single-run comparison.

At each boundary, initialize a scalar budget from the Frobenius norm of the
**last applied pre-reset weight update**, recovered from native checkpoint
history at step 40 and from measured applied updates at step 80. Freeze that
budget through the new cycle. After each native Adam proposal, leave a proposal
under budget unchanged; otherwise interpolate all A/B parameter steps by one
common positive scalar. The budget ratio is one, not a tuned coefficient.

## Exact factor-product accounting

Scaling both parameter updates changes their product quadratically. A simple
`budget / proposed_weight_norm` scaling need not enforce the desired bound.
The implementation accounts for the cross term without constructing dense
weight matrices:

```python
da = proposed_a - old_a
db = proposed_b - old_b
linear = scale * (db @ old_a + old_b @ da)
quadratic = scale * (db @ da)
# Applied weight update: t * linear + t**2 * quadratic
# Norms combine every adapted layer. Use a conservative triangle bound:
t = min(1.0, 2 * budget / (
    linear_norm + (linear_norm**2 + 4 * quadratic_norm * budget)**0.5
))
a.copy_(old_a + t * da)
b.copy_(old_b + t * db)
```

The dense expressions above explain the algebra; production evaluates norms
using rank-one factor inner products. After FP32 parameter writes, it measures
the actual applied update and fails if the bound is exceeded beyond roundoff.
The bound is conservative, not necessarily the largest permitted step.

Adam's gradient moments and bias-correction counters remain those produced by
the native optimizer. Interpolation changes only the applied parameter step,
equivalent to a common reduction of the effective LR for that update. The
constant scheduler itself is unchanged; `update_cap/step_multiplier` records
this extra control separately. Fresh adapters still learn independent
directions, but useful accumulated rank and performance are not guaranteed.

## Measurements and execution

Record proposed/applied weight-update norms, budget and multiplier, reward,
entropy, gradient norm, and accumulated singular spectra/stable rank. Final
rank is measured after interpolation. Evaluate AIME25 avg@8/pass@8 every
20 steps and sample all-GPU VRAM every 10 seconds. Additional allocator-peak
records include the cap's factor gathers and parameter writes.

Tests verify the dense norm oracle, cross-term accounting, a combined
multi-layer bound, unchanged under-budget steps, invalid-budget rejection,
and preservation of real Adam moments/counters and the scheduler while applying
the cap. The production recipe passes SkyRL validation and Ruff.

The user authorized a delayed launch: **not before October 6, 2026, 13:58:20
UTC**, and only after the current three-branch comparison finishes successfully
with all step-100 evaluations present. The queue also waits for all eight GPUs
to be free and checks source/config fingerprints before launching. It stops
on failure rather than substituting a checkpoint or interrupting another run.

**Results are pending.** The hypothesis is that controlling the reset-induced
step-size jump improves continuity while preserving the rank-growth mechanism.
