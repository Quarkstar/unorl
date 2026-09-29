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

## UNORL trial: NoRA-init with rank-one GRPO

Profile: `configs/qwen3-4b-base-grpo-lora-r1-nora-init.json`.
The reference is the full-layer rank-one GRPO blog recipe. This trial changes
only initialization (`nora_init`) and alpha (32 to 1). It evaluates the
normalized initialization with the paper's recommended unit effective scale;
it cannot isolate initialization from the alpha change. AdamW LR stays
1.5e-5, with no warmup; data, prompts, eight rollouts, 32 prompts per step,
8K response cap, 100 steps, eight GPUs, and AIME25 evaluation every 20 steps
all match the reference.

`unorl/nora.py` normalizes the default Kaiming A columns once before FSDP
sharding and optimizer creation. At rank one this retains each random sign,
so the ±1 fraction is approximately, rather than exactly, 50/50. B remains
zero. A and B both train freely afterward; there is no continuing constraint,
SGD, adapter merging, layer restriction, or quantization in this experiment.
The worker records initialization counts in the log. Native PEFT export and
vLLM adapter synchronization remain unchanged. Checkpoint resume happens after
initialization, so restored adapters are not normalized again.

The implementation checks that initialization leaves logits unchanged, that A
receives gradients after B starts learning, and that native adapter reload and
merging preserve trained logits. The completed 100-step run assesses convergence;
this change does not reduce the parameter or optimizer memory.

## Completed trial: acceleration is the positive signal

The [experiment report](../experiments/qwen3-4b-base-grpo-lora-r1-nora-init-20260928-01.md)
contains the full curves, checkpoint evaluations and paired training-window
comparison. The run completed all 100 steps on 2026-09-28 with exit status 0.

NoRA-init reached 33.8% training correctness in steps 21–40 versus 23.7% for
standard LoRA. Step-20 AIME25 avg@8 was 9.2% versus 4.2%. This is promising
faster early learning. Final AIME25 avg@8 / pass@8 was 17.9% / 36.7%, versus
20.0% / 43.3%; the endpoint did not establish a performance gain.

The later plateau is not evidence against NoRA itself. Our next hypothesis
is to merge the adapter into the backbone and reset a fresh NoRA-initialized
adapter, allowing accumulated weight updates to exceed rank one. Merge/reset
might recover early learning speed, but this is not established by this run.
A future comparison must state the merge interval and optimizer-state reset
policy explicitly and keep the algorithm and other training settings fixed.
No merge follow-up was started when recording these results.
