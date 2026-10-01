# LoRA training: 2025–2026 investigation

Reviewed **2026-09-29**. Scope: work first released in 2025–2026, plus the
2026 revision of LoRA-FA. This is a selected review of methods relevant to
UNORL, not an exhaustive bibliography. Versioned papers and inspected upstream
code are distinguished from our own deductions and proposed experiments.

## Recommendation

**LoRA-FA is the first candidate for reducing adapter activation memory;
LoFT is the stronger candidate for studying optimizer continuity.** Both
correct low-rank updates, but they solve different parts of our problem.
Neither currently establishes stable rank-one, long-sequence, on-policy RL
on a phone. Test them separately before combining them with merge/reset.

Our two questions remain:

1. Can successive adapters learn useful new directions after merging?
2. Can we change adapter coordinates without losing useful Adam history?

The existing [NoRA investigation](nora.md) and completed experiment pages
provide local evidence. The current last-half NoRA-init run is a separate
experiment; this literature review changes no training configuration.

## Map of the methods

| Method / reviewed version | Main mechanism | Potential memory benefit | Merge/reset continuity | Rank-one relevance |
|---|---|---|---|---|
| LoRA-FA, May 2026 revision; originated 2023 | Freeze A, correct B gradient | Remove wide inputs saved specifically for A gradients | Fixed A avoids moving coordinates within a cycle; reset still needs a rule | Fixed input direction is a strong restriction |
| LoFT, March 2026 revision; originated 2025, ICLR 2026 | Alternate factors; calibrate gradients and Adam moments | Small persistent state at low rank; dense temporary buffers can increase peak | Explicit moment calibration for evolving factors | Paper includes rank-one supervised results |
| NoRA, August 2026 | Normalize A columns, continuously or at initialization | No inherent activation reduction | Does not specify an Adam reset solution | Initialization becomes random signs; already tested locally |
| PoLAR, June 2025 | Separate orthogonal directions and scale | No demonstrated saving for our workload | No established merge/reset protocol | Most capacity-utilization motivation concerns rank above one |
| LoRA-Pre, February 2026 | Factorize optimizer momentum | Save state for large trainable weight matrices | Useful optimizer-design reference | Different from a rank-one adapter |
| SLAO / Merge before Forget, December 2025 | Reuse and asymmetrically merge one adapter across tasks | Bounded adapter storage across tasks | Parameter reuse, not verified Adam-state transport | Retained adapter still has its chosen rank |
| No Subspace to Track, July 2026 | Diagnose basis noise; transport optimizer state | Analysis of compressed optimizers | Direct warning against blindly carrying moments | Does not evaluate our rank-one RL setting |

Each row is expanded and sourced below. Memory benefits refer to a particular
allocation, not automatically to total process peak memory.

## LoRA-FA: activation saving through a fixed projection

The [2026 paper revision](https://arxiv.org/html/2308.03303v3) freezes the
input projection A and trains B. B backward needs the narrow projected
activation; A backward would need the wide input. It additionally corrects
B's gradient using an inverse Gram matrix, minimizing the discrepancy from
the full gradient within the available subspace. Simply freezing A is an
ablation, not the full corrected method. Its reported supervised benchmarks
do not demonstrate rank-one RL. This method originated in August 2023;
April/May 2026 revisions are the reason it appears in this review.

### Our interpretation and implementation implications

Use the PEFT orientation consistently:

```python
# A: [rank, in_features], B: [out_features, rank]
scale = lora_alpha / rank
adapter_output = scale * (x @ A.T) @ B.T
A.requires_grad_(False)
```

Freezing A does **not** mean detaching `x @ A.T`: upstream trainable layers
still require its input gradient. Attention, nonlinearities, checkpoint
boundaries and other branches can retain the same wide input. Therefore the
paper's single-linear-layer saving is not a whole-transformer memory estimate.
Our existing checkpointing can also reduce the additional saving.

The corrected gradient in this orientation is:

```python
# Conceptual FP32 calculation, not a complete optimizer.
gram = A.float() @ A.float().T
corrected_grad_B = (B.grad.float() @ torch.linalg.pinv(gram)) / scale**2
```

The [PEFT implementation inspected at commit b8674c8](https://github.com/huggingface/peft/blob/b8674c86183a5dee38d0c3ede392e189593025e5/src/peft/optimizers/lorafa.py)
freezes A, regularizes the Gram inverse, corrects the gradient before updating
Adam moments, and exposes `create_lorafa_optimizer`. SkyRL integration must
preserve paired A/B access under FSDP; having a PEFT helper does not establish
compatibility with sharded parameter shapes.

**Important rank-one deduction:** the Gram matrix is a single scalar. With a
fixed A, its correction is a constant positive multiplier on each module's
B gradient. Adam largely cancels a constant multiplier between its first
moment and square-root second moment, except for epsilon, clipping, numerical
precision and other implementation effects. Do not expect the correction
alone to unlock new directions. A fixed rank-one A also restricts each row
of the weight update to that input direction; it cannot represent arbitrary
rank-one updates with another input direction. Broad expressivity claims
should not be read as such a guarantee.

## LoFT: gradient geometry plus optimizer geometry

[LoFT v2](https://arxiv.org/html/2505.21289v2), initially May 2025 and accepted
at ICLR 2026, alternates A/B updates, rescales gradients, calibrates first
and second moments, and aligns clipping with projected full-weight updates.
Alternation removes the simultaneous-factor cross term. Calibration adjusts
history as the opposite factor changes. Its full-rank equivalence to AdamW
and matrix-factorization analysis are conditional mathematical results, not
rank-one RL guarantees. It includes supervised rank-one experiments.

Appendix E.2 reports LLaMA-7B memory of 28.50 GB for rank-16 LoRA versus
35.81 GB for LoFT. “LoFT (simple)” omits second-moment calibration: 30.02 GB
with about 0.1 percentage-point accuracy loss in that comparison. These
measurements are not predictions for UNORL.

### What the reference code actually allocates

Inspected [optimizer](https://github.com/tnurbek/loft/blob/148998d98f901cd7744db7baa5b2b4868fa16bf0/loft_optim/optimizer.py)
and [helpers](https://github.com/tnurbek/loft/blob/148998d98f901cd7744db7baa5b2b4868fa16bf0/loft_optim/optim_helper.py)
at commit `148998d`:

- Persistent second-moment cross terms use arrays shaped `[width, r, r]`.
- Previous factors are retained for moment transport.
- Full calibration reconstructs a dense weight-shaped update and denominator
  before projecting back. Small persistent state does not imply small peak.
- The simpler branch avoids that dense reconstruction in the optimizer;
  clipping and the complete execution path still need a memory audit.

For illustration, a Qwen3-4B MLP matrix `[9728, 2560]` has about 24.9 million
entries. **One FP32 dense temporary is about 95 MiB**, independently of rank.
Several simultaneous buffers or FSDP parameter gathering can add more. This
is an allocation estimate, not a measured training peak.

LoFT's moving-coordinate treatment is relevant to merge/reset, but its
published algorithm does not automatically define a function-preserving
reset with both new B=0 and transported states. That boundary needs its own
algorithm and validation. Alternating factors also does not automatically
save A-gradient activations: skipping an optimizer update after backward
leaves the backward storage requirement intact.

## Are LoFT and LoRA-FA similar?

**Yes: they share inverse-Gram correction of the induced weight update.**
For a fixed A, plain gradient descent on B produces a weight update filtered
through `A.T @ A`. The following comparison is our chain-rule derivation:

```python
# G is the conceptual full-weight gradient; never materialize it in training.
grad_B = scale * G @ A.T
plain_weight_step = -lr * scale * grad_B @ A

corrected_grad_B = grad_B @ torch.linalg.pinv(A @ A.T) / scale**2
corrected_weight_step = -lr * scale * corrected_grad_B @ A
# corrected_weight_step = -lr * G @ P
# P projects onto A's row space.
```

Projection corrects geometry **inside the available subspace**. It does not
restore gradient components outside it, and this plain-gradient calculation
is not an exact description of an elementwise Adam step.

The practical difference is that FA keeps that subspace fixed to save
activations. LoFT lets both factors evolve and handles the resulting
optimizer-history mismatch. Consequently, a future combined approach might
freeze A during each cycle and transport history when refreshing it. This
is a research hypothesis, not an already validated combination.

## Other recent work worth learning from

### NoRA: conditioning at initialization

[NoRA, August 2026](https://arxiv.org/html/2608.31036v1) normalizes A columns.
NoRA-init performs this once, then trains ordinary adapters. At rank one,
initialization gives random signs; continuously enforcing unit norm instead
makes each scalar a sign with zero derivative away from zero. That latter
observation is our differentiation argument, not an RL result from the paper.
We use initialization-only NoRA. See the [local method and trial record](nora.md).
It addresses update conditioning, not accumulated rank or reset history.

### PoLAR: measure useful rank rather than nominal rank

[PoLAR, June 2025](https://arxiv.org/html/2506.03133v1) separates two orthogonal
direction factors from a scale matrix. It targets the observation that a
nominally high-rank LoRA update can have low stable rank, with conditional
matrix-factorization theory and supervised benchmark evidence. Its reported
speed of convergence is not a general RL rate guarantee.

Our implication: after multiple merges, measure the singular-value spectrum
of the **accumulated** update. Rank-one adapters individually have stable
rank one when nonzero; the interesting question is whether their sum develops
several substantial singular values. Enforcing new A orthogonality alone
cannot guarantee this if the learned B directions collapse together.

### LoRA-Pre: compressed optimizer state is a different target

[LoRA-Pre, February 2026](https://arxiv.org/html/2602.24283v1) interprets
momentum through online regression and factorizes the momentum matrix.
Experiments include language-model pretraining and supervised math adaptation.
It saves optimizer state for large trainable matrices; it is not equivalent
to freezing a backbone and training tiny adapters.

For UNORL, optimizer-state compression has lower priority: rank-one Adam
state is already small relative to weights and long-sequence activations.
Audit full-gradient buffers before adopting this family for a phone.

### Merge before Forget: bounded storage, not unbounded accumulated rank

[SLAO, December 2025](https://arxiv.org/html/2512.23017v1) reuses one LoRA
across sequential tasks, extracts an orthogonal row basis for A, reuses B,
and asymmetrically merges B with time-aware scaling. Its analysis concerns
continual learning and uses assumptions on SGD dynamics. Orthogonal rows
within A are not a guarantee of orthogonality to every historical adapter.
The retained product remains rank at most r; this is different from merging
successive low-rank products into a full base matrix. It does not specify
preservation of Adam moments. Learn from its parameter reuse, but do not
claim it solves our two problems together.

### No Subspace to Track: stale moments are a concrete failure mode

[July 2026 preprint](https://arxiv.org/html/2607.05872v1) studies refreshed
low-rank gradient optimizers. Much observed basis rotation can reflect
minibatch noise rather than a stable changing signal. Transporting state
helps; lowering beta2 can help in tested recipes, with explicit recipe and
training-length caveats. This is evidence from compressed full-weight
optimizers, not our adapter RL. The earlier
[LDAdam paper](https://arxiv.org/html/2410.16103v3) is relevant background
outside the review window: projection-aware transport and error feedback.

Our implication: use factor overlap and optimizer diagnostics, rather than
blindly copying moment tensors into unrelated adapter coordinates. A smaller
beta2 is a separate controlled ablation, not a substitute for transport.

## Rank growth and continuity: what can be preserved?

This section is our proposed analysis, not a published combined method.
Merging `scale * B @ A` into W and replacing the adapter with B=0 can preserve
the model function, up to arithmetic precision. It does **not** preserve the
optimizer's coordinate system or its next update.

For a normalized fixed rank-one A, the part of old B momentum representable
in a new A direction has a simple projection:

```python
# q_old, q_new: unit input-space vectors; m_B_old: output-space vector.
overlap = torch.dot(q_new, q_old)
m_B_new = overlap * m_B_old
```

This transports the old **represented** weight-space momentum, not a full
historical gradient that we never stored. If the directions are strictly
orthogonal, `overlap == 0`: none survives in that new one-dimensional input
space. Projecting a stored second moment needs care about cross terms and
unseen directions; copying it unchanged has no general justification.

Two possible compromises:

- **Gradual refresh:** mix the old direction with a new orthogonal component,
  normalize, and transport the representable history. This trades immediate
  direction novelty for overlap; it is not a guarantee of useful rank growth.
- **Overlapping adapters:** add a zero-output new adapter while retaining the
  old adapter and its Adam state for a transition. Later freeze and merge the
  old one. The new adapter keeps its own history, at temporary extra memory
  cost. This preserves function at introduction, but does not preserve all
  retired history forever.

A scheduler that reduces LR to zero near a reset remains a simple control.
It masks a discontinuous update boundary rather than restoring lost history.

## Proposed experiments, one change at a time

LoRA-FA and the fresh-state NoRA merge/reset trial have completed; LoFT-simple
is running as of 2026-10-01. Other comparison arms below remain plans.
Keep the selected GRPO recipe,
model, dataset, eight rollouts, response budget, evaluation and Adam-family
settings fixed except for method-required changes. Record every change;
matched numeric LR need not mean matched effective weight-space steps.

1. **LoRA-FA:** compare trainable A/B, frozen A with ordinary AdamW, and frozen
   A with the corrected optimizer. Match A initialization and alpha within
   this comparison. Audit FSDP pairing and initial logits first. Measure both
   learning and memory; rank-one success is not assumed.
2. **LoFT (simple), then full LoFT if justified:** separate optimizer geometry
   from factor freezing. Record effective scale, alternation, epsilon,
   clipping and moment settings. Match rollout compute, not just update count.
3. **Merge/reset comparison:** only after a reliable non-merging reference,
   compare fresh state, gradual refresh with transport, and an overlap
   transition. Do not simultaneously change layer placement or RL algorithm.

Required observations: training correctness, AIME25 avg@8/pass@8, entropy,
response length, clipping fraction, gradient norm, actual merged-weight
update norm, A/B norms, factor overlap, and moment norms around boundaries.
For rank growth, record accumulated-update stable rank and retained spectral
energy using a low-rank factor representation where practical, avoiding a
permanent dense copy of each layer's update.

Keep the existing **10-second external VRAM sampler**, and add training-phase
allocator high-water marks when implementing a method. External samples can
miss short optimizer workspace peaks. Separate persistent state bytes,
activation storage and transient buffers; include inference residency in
end-to-end peak reporting. NVIDIA measurements evaluate this implementation,
not iPhone feasibility. Phone validation must eventually account for unified
memory and the actual supported numerical kernels.

## Implemented LoRA-FA comparison: 2026-09-30

Profile: `configs/qwen3-4b-base-grpo-lorafa-r1.json`. All layers, rank one,
Kaiming A, zero B, alpha 32, native AdamW LR 1.5e-5, constant schedule without
warmup, 32 prompts × eight rollouts, 8K response cap, 100 steps, eight A100s,
and AIME25 avg@8/pass@8 every 20 steps. The reference is the standard full-layer
rank-one GRPO recipe. `lora.init_method=lorafa` selects the custom worker but
is translated to Kaiming before PEFT initialization; it does not introduce
NoRA initialization or another scaling change.

`unorl/lorafa.py` freezes A before sharding. `unorl/lorafa_worker.py` retains
SkyRL's native FSDP2 strategy and AdamW. After initialization, each A is
materialized once to compute its FP32 squared norm. The cached correction is
`1 / (scale**2 * (A.square().sum() + 1e-8))`. It multiplies accumulated B
DTensor gradients locally before native gradient clipping, AdamW and scheduler
stepping. Checkpoint loading rebuilds the cache from restored A. This
rank-one specialization needs no dense weight-shaped correction tensor.

This is the paper's gradient correction with the **reference's native AdamW
implementation and defaults** retained. It does not reproduce every numerical
default or epsilon convention of PEFT's separate optimizer. The recorded
`policy/grad_norm` is measured after correction and before clipping, so its
magnitude is not directly comparable with the reference's uncorrected norm.
The initialization audit records correction factors and optimizer settings.

Expected trainable B count for this architecture: **1,105,920 parameters**;
A's 958,464 parameters remain stored but frozen. All backbone parameters stay
frozen. A/B export and vLLM synchronization use native PEFT paths.

The worker records per-rank CUDA allocator peak allocated/reserved bytes from
policy forward/backward through the optimizer step. These include the resident
policy allocations in that process; weight backloading before forward/backward
is outside the reset window. The external 10-second NVML sampler additionally
covers device residency across rollout, evaluation and training. No matched
full-layer reference allocator measurement exists yet, so measured peaks alone
cannot establish the size of the saving against that old run.

Correctness checks cover projected-gradient geometry, unchanged initial
logits, frozen A/backbone, B-only Adam state, activation storage in an isolated
adapter branch, and native adapter reload/merge. A distributed tiny-model check
uses the actual SkyRL strategy on the intended GPU count and verifies
checkpoint/resume. Smoke artifacts are excluded from published experiment
results. Learning and end-to-end memory benefits remain experimental questions.


## LoFT-simple implementation and launch: 2026-10-01

Run: `qwen3-4b-base-grpo-loft-simple-r1-20261001-01`.
[Experiment page](../experiments/qwen3-4b-base-grpo-loft-simple-r1-20261001-01.md).
Profile: `configs/qwen3-4b-base-grpo-loft-simple-r1.json`.

`unorl/loft.py` specializes the pinned authors' LoFT-simple implementation to
rank one. It uses ordinary Kaiming A / zero B, **alpha=1**, epsilon **1e-4**,
betas 0.9/0.999 and zero weight decay. Keep LR 1.5e-5, 100 updates, all
36 layers, 32 prompts × eight responses, an 8192-token cap, all eight A100s,
and AIME25 avg@8/pass@8 every 20 steps. Relative to standard LoRA, alpha and
epsilon change with the optimizer method; this is not an isolated optimizer
ablation with identical numerical scaling. No NoRA initialization, frozen A,
merge/reset, SGD or additional warmup is introduced.

The active factor alternates B, A, B, A. Both factors' first and second
moments update every step; first moments are transported using the overlap
with the saved previous opposite factor. Second moments stay elementwise and
are not transported. Regularization is 1e-6 for A's opposite-factor Gram and
1e-8 for B's. Clipping follows the authors' **simple** branch: compute the
norm of active calibrated factor gradients, then scale all raw gradients.
It does not use the full variant's dense projected-weight norm. Recorded
`policy/grad_norm` therefore differs in meaning from the standard LoRA norm.

For FSDP2, gather only small rank-one factors/gradients for scalar calibration
and update local shards in place. No dense weight-shaped calibration buffer
is constructed. Previous opposite factors are saved inside optimizer state,
and the alternating update phase is included in `state_dict`; this repairs
the reference optimizer's otherwise unsaved auxiliary history for our resume
contract. Native PEFT adapter synchronization and export remain in use.
A/B still participate in backward, so this does not claim LoRA-FA's activation
saving. Training allocator peaks and ten-second NVML readings are recorded.

Validation: 77 tests passed. `scripts/check_loft_reference.py` compared eight
CPU updates against the pinned authors' implementation including clipping;
maximum parameter difference was 1.49e-8. `scripts/check_loft_fsdp.py` verified
actual eight-GPU FSDP2 updates, a frozen backbone, native A/B export and exact
next-update reproduction after checkpoint reload. The tiny model's calibrated
A norms exceeded 100 before clipping, which is permitted by the algorithm;
validation checks finite gradients and resume correctness rather than applying
a bound copied from ordinary LoRA. Learning benefit remains unmeasured.


## Main research line: make ReLoRA sustain RL learning (2026-10-01)

The project priority is repeated rank-one merge/restart training. Standalone
LoFT and quantization are secondary until this mechanism is understood. The
success criterion is better sustained learning than a matched non-merging
rank-one reference, while keeping only one active adapter and bounded memory.

Our completed NoRA merge trial tested constant LR plus complete Adam-history
removal. It did not reproduce the stabilization components of the original
[ReLoRA paper](https://arxiv.org/html/2307.05695v2): partial moment reset and a
restart LR schedule with warmup. The paper describes pruning 99% of small
state entries. The current [official repository](https://github.com/Guitaricet/relora)
at `176f37633fe02019835387258ddabcf6d91e328d` offers several reset modes;
its README recommends 90% magnitude pruning or its nearly complete reset
mode. These settings must be distinguished rather than called one recipe.
The repository prunes moment tensors in place rather than deleting all state
and resetting Adam's bias-correction counters as our earlier worker did.

The earlier run does not prove an immediate reset-induced collapse. Training
correctness averaged 37.0% at steps 31–40, 39.2% at 41–50, 38.7% at 71–80,
41.0% at 81–90, then 33.4% at 91–100. Boundary discontinuity, insufficient
useful rank growth and unrelated late training dynamics remain separate
hypotheses. The four-token logit probe is inadequate to establish an RL
impact; validate log-probability/KL changes on actual math trajectories.

Controlled restart comparison (2026-10-01): implementation and validation completed; the five-update ramp runs first, followed by the constant-LR control on successful completion. Both use native AdamW with full state clearing, merges at 40 and 80, and the unchanged standard-LoRA GRPO settings. Actual eight-GPU FSDP2 checks cover merging, restart scheduling, response-prefix probes, checkpoint loading and dense export. CPU checks cover the fixed-A preserved-history control and low-rank spectra. Each boundary probes up to 128 real response tokens per rank and reports token-weighted KL/log-probability changes across all ranks. Per-layer accumulated-update singular values are saved at steps 40, 80 and 100. Training allocator peaks and ten-second NVML samples are recorded. Full Adam reset is held constant to isolate the ramp; this pair is not the complete published ReLoRA recipe.

Follow-up sequence:

1. Establish merge/sync correctness on fixed math trajectories and measure
   accumulated-update singular values from stored low-rank factors. A fixed-A,
   B-only, no-decay control with preserved Adam history checks the expected
   reparameterization equivalence before introducing new directions.
2. Use the working standard-LoRA GRPO recipe as the reference: Kaiming A,
   alpha 32, rank one, AdamW, unchanged data/rollouts/batch/response budget.
   Add repeated merges and a brief post-reset LR ramp; compare with the same
   merge setup without the ramp. Initial training warmup stays unchanged.
3. Compare full history removal with partial moment pruning, keeping the
   schedule fixed. Retain Adam counters for pruning; log retained moment
   energy and effective weight-space step sizes. This tests the original
   stabilization idea, not exact optimizer-coordinate transport.
4. If useful rank still does not grow, test gradual A refresh with explicit
   representable-history transport. This is a later proposed extension,
   separated from the original-style restart recipe.

A longer matched non-merging control is needed before claiming improved
learning beyond a rank-one ceiling. Two merges in 100 steps are preliminary
evidence. Do not change initialization, adapter placement, RL algorithm,
precision or quantization together with restart behavior.
