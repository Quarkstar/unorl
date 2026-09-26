# NoRA: method notes

Paper: Jiale Kang et al., [Normalized Low-Rank Adaptation](https://arxiv.org/abs/2608.31036), arXiv v1, 2026-08-31.

## Method

For a LoRA layer with down-projection `A ∈ R^(r×k)` and up-projection
`B ∈ R^(d×r)`, standard LoRA adds `αBAx`, with `B=0` at initialization.
Therefore the model initially stays exactly at the base weights, and at the
first update `A` receives zero gradient while the gradient for `B` depends on
`A`.

NoRA treats each column `a_j = A[:, j]` as the projection direction for one
input coordinate and normalizes each column along the rank dimension:

`Norm(A)[:, j] = A[:, j] / max(||A[:, j]||₂, ε)`.

The constrained NoRA forward update is `α B Norm(A) x`; it normalizes columns
throughout training. Since normalization is parameter-only, the layer remains
linear in `x` and the resulting update can be merged into the base weight.
**NoRA-init** applies this normalization only once, sets `B=0`, and then trains
ordinary LoRA. It adds no trainable parameters or continuing normalization
overhead.

## Why it may help

At initialization, a gradient step on `B` induces an approximate merged-weight
step `ΔW = -η G P`, where `G` is the full-weight gradient and
`P = α² AᵀA`. The diagonal entry for input coordinate `j` is
`α² ||a_j||²`; this acts like that coordinate's effective learning-rate scale.
Standard random LoRA initialization can make these diagonal values small and
uneven. NoRA makes every column norm one, so `diag(P)=α² I` and the random
projection no longer assigns different scales to input coordinates. `P` is
still low-rank; NoRA does not turn rank-one LoRA into full fine-tuning.

## Rank-one interpretation

When `r=1`, each column `A[:, j]` is a scalar. Unit-norm normalization maps a
nonzero scalar to its sign. With a symmetric continuous random initialization,
NoRA-init therefore gives each A entry `+1` or `-1`, approximately 50/50.
This matches the proposed rank-one ±1 initialization exactly; it is the
rank-one form of **NoRA-init**. For higher ranks, NoRA does not mean independent
±1 entries: each length-r column is normalized to unit Euclidean norm.

The paper's parameterization writes the update as `αBA`; with the common
library convention `lora_alpha/r`, its recommended `α=1` corresponds to
`lora_alpha=r`. For rank one that means `lora_alpha=1`, while the current
rank-one baseline uses `lora_alpha=32`. Combining normalized `A` with alpha 32
would substantially change the update scale. A NoRA experiment must account
for this convention explicitly and log the effective scale; it should not
silently reuse the current alpha.

## Evidence and limits

The paper's RLVR study uses DAPO on DeepSeek-R1-Distill-Qwen-1.5B and
DAPO-Math-17K: rank 32, alpha 64, eight responses per prompt, global batch
128, and 1,024 steps. Its table reports an aggregate score of 44.4 for NoRA
versus 42.8 for LoRA. This supports the normalized projection in an RLVR
setting, but it is not the same as our rank-one, single-rollout REINFORCE run.
The RL table compares NoRA with LoRA; it does not report NoRA-init separately.

For our planned test, the closest simple implementation is initialization-only
NoRA: initialize the usual A randomly, normalize each A column to unit norm,
keep B zero, and then use standard LoRA training. At rank one this is
equivalent to sign initialization. The α/r scale and learning-rate comparison
must be recorded as part of the method.
