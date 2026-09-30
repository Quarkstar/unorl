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

## Follow-up: NoRA-init in the last half of the network

Profile: `configs/qwen3-4b-base-grpo-lora-r1-nora-init-last18.json`.
Run ID: `qwen3-4b-base-grpo-lora-r1-nora-init-last18-20260929-01`.

The hypothesis is that NoRA-init shortens the slower early learning observed
with adapters only in layers 18–35. Relative to full-layer NoRA-init, only the
layer exclusion changes. Relative to standard last-half LoRA, initialization
and alpha change together (Kaiming/32 to NoRA-init/1).

The model, data, prompts, AdamW LR 1.5e-5, 32 prompts × 8 rollouts, 8K response
limit, 100 steps and eight GPUs match the earlier GRPO experiments. Evaluate
AIME25 with eight samples per question before training and every 20 steps.
No merge/reset is applied: first test whether normalized initialization alone
with its unit scale can address the delayed start. This uses the existing
adapter-placement implementation; a frozen-prefix activation cutoff is not
part of this experiment. No learning conclusion is recorded before results.


For the last-half NoRA run, `scripts/monitor_vram.py` attaches an external
NVML sampler at 10-second intervals before the first training update. It
writes `vram-samples.jsonl` (per-device and per-process readings) and
`vram-summary.json` (running per-device peaks grouped by rollout, evaluation,
training-forward and training-update phases). Device readings include all
resident processes, including inference allocations. Phase labels follow
console timer events and can lag because of buffering. These are sampled
peaks, not exact PyTorch allocator high-water marks; sub-interval spikes and
startup before attachment may be missed. The sampler exits automatically when
the run's exit-status file appears.

The sampler initially used 0.5 seconds, then switched to 10 seconds at the user's request before training began. Earlier samples and peaks are retained.


## Full-layer merge/reset follow-up (2026-09-30)

Selected follow-up: full NoRA-init with merges after optimizer updates **40 and
80** in a 100-step GRPO run. Profile:
`configs/qwen3-4b-base-grpo-nora-merge-r1.json`; entrypoint:
`unorl.nora_merge_train`. The profile differs from the completed full-layer
NoRA-init reference only by the added merge interval. Keep Qwen3-4B-Base,
all 36 layers, rank 1 / alpha 1, AdamW LR 1.5e-5, no warmup, 32 prompts ×
8 responses, 8192 response tokens, eight GPUs and AIME25 avg@8 / pass@8
before training and every 20 steps. LoFT-simple remains a separate candidate.

At each boundary, accumulate the current scaled B @ A into the frozen FP32
backbone, initialize a fresh A with independent ±1 entries, and zero B.
Parameter objects and scheduler state are preserved. **Clear all adapter AdamW
moments and step counters**; the next update starts fresh optimizer history at
the same constant LR. This is an explicit reset baseline, not a method that
solves optimizer continuity. After two merges, each accumulated weight update
can have rank at most three; neither high effective rank nor improved learning
is guaranteed.

Rollout retains the reference's native rank-one vLLM adapter path. At startup,
resume and after each merge, send accumulated backbone W over NCCL first,
then load the current adapter through SkyRL's native LoRA synchronization.
Other updates synchronize only the adapter. Sending only the adapter after a
merge would lose prior learned updates; sending W + B @ A and also loading
the adapter would count the active update twice. Final HF export materializes
W + B @ A; FSDP checkpoints retain the merged backbone, active factors and
optimizer/scheduler for resumption.

Validation: 72 CPU/runtime tests passed, including parameter identity,
optimizer reset/repopulation, profile equality and synchronization order.
An eight-GPU actual FSDP2 check completed four AdamW updates with a merge
between updates two and three, then verified checkpoint reload and dense
export. The tiny BF16 forward probe changed by a maximum 0.00114 in logits;
merging preserves the function in exact arithmetic, but mixed-precision
forward computation has rounding differences. The full run records this probe
at each merge rather than assuming bitwise equality.

Record exact PyTorch peak allocated/reserved memory for each training update,
including merge work at boundaries. Sample NVML memory every **10 seconds**
for rollout, evaluation, training and synchronization. Merge gathers may raise
peak memory. Both measurements are required to distinguish training allocator
memory from total device residency. Results remain pending until measured;
compare the post-40 and post-80 slopes and AIME25 checkpoints against the
completed full-layer NoRA-init reference.
