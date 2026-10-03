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


### October 2: diagnostic failure and restart

The first standard-LoRA restart-ramp attempt completed 39 updates and failed
while preparing the step-40 probe, before any merge. SkyRL left-pads complete
trajectories and right-aligns response indicators. For a shorter response, the
prompt can fall inside the common response slice. The diagnostic now aligns
response indicators to full sequence positions and extracts prompt tokens
using attention AND NOT response position. It also skips empty padded rows.
No reward, loss, optimizer or schedule setting changed.

A regression test uses native SkyRL preprocessing. Eight-GPU FSDP2 validation
now invokes the actual ReLoRA worker boundary method, checking collective
response-prefix diagnostics, accumulated rank, reset scheduling, checkpoint
loading with exact next-update reproduction, and dense export. All 85 tests
passed. The pair restarted from the base model under the `20261002-01` suffix;
no checkpoint before step 100 existed for the failed attempt. Its logs and
evaluation records are retained. Before merging, reward increased normally;
no conclusion about the benefit of ReLoRA follows from that attempt.


### October 3: controlled restart comparison completed

Both restarted runs finished 100 updates with successful merges at 40 and 80.
Final training correctness over steps 81–100: ramp 36.31%, constant resets
37.03%, standard LoRA 37.48%. Final AIME25 avg@8 / pass@8: ramp 15.83% /
40.0%, constant resets 17.08% / 30.0%, standard LoRA 20.0% / 43.33%.
There is no clear benefit from this five-update restart ramp. The initial
conditions and sampling already differ slightly across runs, so small
evaluation differences on 30 questions do not establish a reliable ranking.

Accumulated-update mean stable rank at step 100 was 1.887 for ramp and 2.160
for constant resets. Mean energy outside the leading direction was 46.48%
and 53.25%. Rank growth is happening, but it does not imply useful policy
improvement. Mean response-prefix KL at merge boundaries was 0.00054–0.00057,
with 1024 probed tokens per boundary. No immediate post-merge reward collapse
appeared. Both pairs and the non-merging reference fell in reward in the last
ten updates, so that late fall alone is not evidence of reset damage.

The next controlled candidate is partial moment pruning with Adam counters
retained, compared against the constant-LR full-state-clear recipe. Hold
initialization, rollout count, schedule, batch and merge interval fixed. This
is proposed, not launched. A longer matched non-merging control remains needed
to investigate rank-one ceilings.

Both runs' large weights and factor caches were cleaned up. The cleanup script
also mistakenly removed raw evaluation response dumps beneath exports. Aggregate
evaluation records, logs, metrics, memory traces and rank diagnostics are
retained; per-response regrading is unavailable. Cleanup now removes only
checkpoint/state files and numeric global-step policy/critic weight directories.
A filesystem regression test verifies evaluation and diagnostic retention.


### Where the merge curves diverge from standard LoRA

![Aligned reward, entropy, response length, gradient and evaluation curves](../figures/relora-boundary-analysis.svg)

The plot uses disjoint five-step means; raw values are faint. Merge markers
are at 40.5/80.5 because step-40/80 training reward was sampled **before** that
step's optimizer update and merge. The step-40/80 evaluation is **after**
the update/merge. Step-41/81 reward is the first rollout under the merged
policy, before its first fresh-adapter optimizer update. Mixing these
conventions would make an apparent boundary drop misleading.

| Steps | Standard correctness | Merge + ramp | Merge + constant LR | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 1–20 | 12.25% | 12.66% | 11.89% | Similar early learning; no merge yet |
| 21–40 | 23.65% | 23.28% | 22.05% | Small differences already exist before intervention |
| 41–45 | 35.00% | 33.20% | 33.59% | No sudden collapse immediately after merging |
| 46–60 | 36.80% | 32.53% | 32.53% | Main divergence: approximately 4.27 percentage points behind |
| 61–80 | 39.26% | 36.35% | 37.32% | Gap narrows, especially without ramp |
| 81–90 | 40.04% | 39.53% | 39.14% | Nearly catches up following second merge |
| 91–100 | 34.92% | 33.09% | 34.92% | Shared late decline; constant-LR merge matches standard |

**A temporary optimization delay is the strongest descriptive reading.**
This is not a persistent lower learning slope or a catastrophic reset.
Linear fits over steps 1–40 are 0.561/0.565/0.517 correctness percentage
points per update for standard/ramp/constant. Over 41–60 they are
0.223/0.163/0.105; over the broader 41–80 interval the merging runs catch up,
with slopes 0.134/0.171 versus standard's 0.100. These noisy local fits depend
on the chosen window and changing question difficulty; they are not causal
estimates or a law of learning. Stepwise reward correlation with standard is
0.957 for ramp and 0.961 for constant resets, consistent with shared batch
variation. Aggregate reward cannot distinguish question hardness from other
shared training dynamics.

Subtracting each run's own pre-merge gap over steps 21–40, the additional
reward deficit over 41–60 is 3.28 percentage points for ramp and 1.95 for
constant resets. This descriptive correction avoids counting the constant
run's existing 1.60-point deficit as a new merge effect; it is not a causal
estimate.

**The accompanying signals:**

| Steps 41–60 mean | Standard | Merge + ramp | Merge + constant LR |
| --- | ---: | ---: | ---: |
| Response tokens | 2,859 | 2,403 | 2,460 |
| Policy entropy | 0.1483 | 0.1696 | 0.1692 |
| Gradient norm before clipping | 0.0350 | 0.0359 | 0.0389 |

Response-length growth falls behind by approximately 399–456 tokens, while
entropy stays higher. This accompanies lower reward; it does not prove that
longer responses would fix accuracy. Gradient norms remain similar or slightly
higher, without an obvious explosion or disappearance. Reported norm is in
adapter coordinates; it does not measure useful weight-space learning.
Rollout/training chosen-token log-probability differences remain near
0.0066–0.0070 in this interval, with no large synchronization mismatch spike.
That does not establish exact merged-policy equivalence.

**The ramp does not explain the whole gap.** The identical constant-LR reset
trial also has the 46–60 deficit. The ramp has a normalized total LR budget
of 95 base-LR update units versus 100 for constant LR: it sacrifices 2.5
units at each restart. This arithmetic is not an equivalence in Adam update
size, but shows that the ramp does consume some optimization budget. It
failed to yield a clear stabilization advantage here.

**Rank is genuinely growing.** Using accumulated singular-value energies
and each cycle's update norm, the inferred global Frobenius cosine between
cycle-1 and cycle-2 updates is about +0.000065 for ramp and −0.000104 for
constant resets. This estimate uses the polarization identity on float32
diagnostics and is approximate. The directions are nearly orthogonal in
aggregate; they are not simply re-adding the same rank-one update. New
directions are not necessarily good task directions.

**The final weight-change magnitude is also comparable.** The saved standard
LoRA step-100 adapter gives global delta L2 3.405 (computed without dense
matrices, as scale times norm(A) times norm(B) per rank-one projection).
Accumulated-factor diagnostics give 3.407 for ramp and 3.607 for constant
resets, excluding tiny FP32 merge-rounding corrections. The merging runs are
not simply ending with a much smaller total weight change. Norms alone cannot
tell whether those changes point in useful directions. The reference's
adapter-file hash and extraction method are saved in
[the portable norm audit](../data/standard-lora-final-update.json).

**What remains unseparated:** each boundary simultaneously merges into the
BF16 forward backbone, freezes the old accumulated update, replaces A with
a fresh Kaiming direction, zeros B, and clears all Adam moments/counters.
Fresh B=0 also starts A's gradient signal at zero. The small but nonzero
response-prefix KL (~0.00055) leaves some policy drift possible; first-128-token
probes cannot exclude long-response effects. Saved scalar metrics do not
identify which mechanism produces the temporary delay. Effective per-update
weight changes and gradient-direction alignment were not logged, and deleted
weights cannot be used to recover them.

**Fairness checks:** both merge profiles preserve the algorithm, model
selection, optimizer LR, rollout count, batch and response budget. The
training-data audit hashes and SkyRL commit match the reference. Absolute
asset paths, project/environment names changed during the repository rename;
the archived and current `score_answer` and `symbolic_score` function ASTs
are identical, and the baseline system-message serialization is unchanged.
Sampling nevertheless produces differences before the first merge. One run
per setting cannot precisely attribute the entire 4.27-point deficit.

The strongest causal comparison would branch the non-merging and merging
policies from the same checkpoint immediately before a boundary, restoring
identical optimizer state, data position and RNG state. Current independent
runs do not share an identical learned policy at step 40. Log effective
weight-space update norms/direction changes through the first 10–20 fresh
adapter updates; scalar adapter gradient norm alone cannot explain the lag.

The next diagnostic should target continuity of useful updates after reset,
not add more warmup or assume that higher rank alone will improve reward.
Partial moment pruning is a controlled candidate, but fresh A changes
optimizer coordinates, so preserving moments is not automatically correct
transport. It remains proposed and untested. A fixed-A/B-only preserved-history
control already passes tiny-model next-update parity; a live version could
isolate merge numerics, but it would be a diagnostic rather than the
full trainable-A method. No new training was launched for this analysis.

Reproduce the numeric windows and plots with
`python scripts/research/analyze_relora.py` after the portable snapshots have
been generated. [Download numerical analysis](../data/relora-boundary-analysis.json).


## First-principles design: gradual A refresh with warm B

### Evidence motivating this trial

The cold-reset runs grow rank but temporarily lose reward progress at steps
46–60; gradient norms do not disappear, and final weight-change norms are
comparable to standard LoRA. A five-update LR ramp does not fix the gap.
The local optimization geometry is a more specific hypothesis than simply
asking for more rank. This is our proposed extension, not a claim that a
published method guarantees success in RL.

[LoRA-Pro](https://arxiv.org/abs/2407.18242) connects the adapter gradients
to the induced low-rank weight update. [ReLoRA](https://arxiv.org/abs/2307.05695)
uses repeated merge/reinitialization with optimizer/schedule stabilization.
Our derivation below targets the discontinuity caused by a cold rank-one
reset, rather than implementing either paper's complete method.

### 1. Why function preservation is insufficient

For a projection matrix, let `G` be the gradient of its effective weight.
For ordinary LoRA with no dropout/DoRA/weight decay:

```python
W_effective = W + scale * B @ A
grad_A = scale * B.T @ G
grad_B = scale * G @ A.T
delta_W_first_order = scale * (delta_B @ A + B @ delta_A)
```

`W += scale * B @ A; B.zero_(); A = fresh_random_A` can preserve the
current effective weight in exact arithmetic, while changing both terms
of its next update. The A gradient becomes zero because B is zero; the B
gradient samples a new input direction. Clearing all Adam state also removes
its previous first/second moments and resets bias correction. Holding the
policy close at the boundary does not preserve its optimization path.

As a diagnostic, for plain SGD with equal factor LR (not our AdamW optimizer),
the first-order induced weight step is:

```python
delta_W = -lr * scale**2 * (G @ A.T @ A + B @ B.T @ G)
```

The cold reset removes the second term and replaces the first projector.
The formula is a local first-order calculation, not a description of Adam.

### 2. Compensated gradual refresh

Keep B nonzero and rotate A by a fixed angle toward a random row direction
orthogonal to the current A. Preserve A's row norm and the existing scale.
The initial trial uses **20 degrees**, chosen before observing its results;
`cos(angle)=0.93969`, `sin(angle)=0.34202`.

```python
# q is orthogonal to A_old and has the same row norm.
A_new = cos(angle) * A_old + sin(angle) * q
W_new = W_old + scale * B_old @ (A_old - A_new)
B_new = B_old
# Therefore, in exact arithmetic:
W_new + scale * B_new @ A_new == W_old + scale * B_old @ A_old
```

This can also be viewed as merging the old adapter, initializing a warm
new adapter, then subtracting that new adapter from the frozen base to
compensate its nonzero initialization. It is a partial merge, not a full
merge followed by zero B. Each correction is rank one, and only one active
rank-one adapter is trainable. No dense optimizer moments or extra learned
modules are added. The native base remains FP32 in storage, BF16 in forward
computation, exactly as in the working recipe.

On the same trajectory in exact arithmetic, `G` is unchanged and B is
unchanged, so `grad_A` is unchanged. B's gradient becomes:

```python
grad_B_new = cos(angle) * grad_B_old + sin(angle) * scale * G @ q.T
```

For normalized rank-one rows, the spectral norm of the input-projector
change is `norm(A)**2 * sin(angle)`. Thus the first-order SGD weight-step
change is bounded by `lr * scale**2 * norm(G) * norm(A)**2 * sin(angle)`.
The B-related output projector is unchanged. This gives a tunable local
discontinuity rather than removing a whole update term. It is not an Adam
convergence theorem or a guarantee of RL reward improvement.

### 3. Adam history: exact parts and approximation

Keep all Adam bias-correction counters. Keep A's first/second moments and
B's second moments. Project B's first moment by the retained row component:

```python
m_A_new = m_A_old
v_A_new = v_A_old
m_B_new = cos(angle) * m_B_old
v_B_new = v_B_old
step_new = step_old
```

A's current gradient continuity is exact in real arithmetic, but the complete
B gradient history along q was never measured. Its mean, variance and cross
terms cannot be reconstructed from old diagonal Adam state. Keeping B's
variance avoids artificially shrinking it by `cos(angle)**2`; it is a
heuristic, not an upper bound on the unknown variance. The first-moment
projection also omits unseen-direction history. We explicitly do **not**
claim exact optimizer transport. Angle zero reduces to the original method
with identical next-update behavior in our control test.

### 4. What rank growth means here

The total effective weight is unchanged immediately at refresh, so its
accumulated update does not instantly gain rank. The frozen compensation
and the newly trained active adapter can acquire different B directions
after subsequent learning, creating higher-rank accumulated updates. If B
directions remain collinear, rank may remain one. We measure the full
compensation history plus the current adapter, not only base-matrix rank.

### 5. Validation completed before launching

- Float64 gradient checks verify exact compensation, unchanged A gradient,
  the rotated B-gradient relation, row norm and rotation angle.
- A warm tiny-model test checks unchanged B, Parameter identities, frozen
  effective weights, preserved Adam counters/variances and projected B moments.
- Continued tiny-model learning creates nonzero update energy outside rank one.
- A zero-angle control reproduces the standard optimizer's next update exactly.
- A factor-only telemetry test agrees with dense weight-update norm and dot
  product without constructing a dense training gradient.
- Actual eight-GPU FSDP2 tests exercise compensation and collective probes,
  native Adam continuation, factor-history checkpointing, exact next-update
  replay after reload, base-only sampler extraction, and dense final export.

Mixed-precision probes remain necessary: exact matrix identities do not
ensure identical BF16 logits. The tiny-model proof is an implementation
check, not evidence of language-model training success.

[Portable validation results](../data/relora-refresh-validation.json) record
94 passing tests and the eight-GPU fixture's exact checkpoint next-update
replay, preserved native Adam history and measured boundary drift.

### 6. Predeclared matched experiment

The old step-40 checkpoint was never retained; the standard-LoRA step-100
checkpoint is available with all eight model, Adam and RNG shards plus
dataloader/trainer state. Both branches start from **that same checkpoint**
and train 100 new updates, global steps **101–200**. The control performs
standard LoRA throughout. The candidate refreshes after **101, 141, 181**.
The first update supplies a real rollout for boundary probes; the initial
evaluation at step 100 occurs before any intervention. Branch sampling
can differ due to nondeterministic kernels/request scheduling, and historical
vLLM sampler RNG is not restored; matching configuration and a common
policy checkpoint do not imply bitwise rollout identity.

Both use full-layer rank one, alpha 32, native AdamW LR 1.5e-5, no initial or
restart warmup, eight rollouts, 32 prompts per update, 8192 response tokens,
no KL, the existing GRPO recipe, all eight A100s, and AIME25 sampled avg@8
and pass@8 every 20 global steps. Resume audits require all Adam counters
to equal the saved scheduler step. Save resumable checkpoints every 20
steps (newest retained) and keep the shared source checkpoint untouched.

Both record effective optimizer weight-step norms and successive-step
cosines, excluding the compensation itself; boundary KL; accumulated spectra;
allocator peaks and ten-second NVML samples. Telemetry uses optional constant
CPU factor buffers, not dense GPU matrices; it can be disabled outside this
research comparison. Compensation factors and previous-update telemetry are
saved in the checkpoint client state, avoiding stale diagnostics after
rollback/resume.

A detached watcher verifies each launcher via `/proc` and records a health
snapshot at startup, hourly, and at terminal state. It records gradient norm,
entropy, response length and recent correctness, and updates each experiment's
book page/plots. Successful exit requires the actual target step; the queued
candidate starts only after the control completes successfully. Failed runs
retain checkpoints for diagnosis. Completion cleans up only weights/state,
preserving evaluation dumps, logs and diagnostic records. Debugging or
method selection still requires this active research agent; the watcher
does not pretend to autonomously reason about and fix arbitrary errors.

**Decision criteria:** an actual nonzero refresh must occur; subsequent
accumulated updates must show useful energy beyond the leading direction;
reward after the intervention should track or improve the matched standard
control, without a persistent AIME25 loss. A single pair on 30 evaluation
questions gives preliminary evidence only. A successful advanced-policy
continuation must be followed by a matched from-base test and replication
before declaring general equivalence or superiority. The goal remains
active until empirical evidence supports the requested performance.

### 7. Predeclared analysis and uncertainty

Compare global-update windows 101–120, 121–140, 141–160, 161–180 and
181–200. Partial windows must show their actual paired-update count and
remain labeled incomplete. The analysis rejects recipe differences beyond
the declared refresh settings and run-specific output paths. Training rewards
use changing on-policy batches, so their window differences are descriptive;
we do not assign independent-sample confidence intervals to serial rewards.

For each shared AIME25 evaluation, pair the two branches by question, retaining
all eight scoring records. Compare per-question correct-response fractions
for avg@8 and the per-question any-correct indicator for pass@8. A deterministic
10,000-draw bootstrap resamples whole questions and reports the 2.5/97.5
percentiles of the mean candidate-minus-control difference. This is a
question-bootstrap interval, not training-seed uncertainty or a guarantee of
equivalence. Shared questions do not imply paired generated trajectories.

Also report each branch's improvement relative to its new step-100 evaluation,
and the difference of those improvements. This descriptive adjustment includes
noise from both starting evaluations; it is not automatically a stronger
causal estimate than the final-score difference. Missing branch/evaluation
records remain missing, and non-eight-sample or mismatched-question records
cannot produce a paired result.

Only hashed prompt identifiers, sample counts and correctness counts enter
the portable continuation snapshots; generated response text stays local.
`scripts/research/analyze_refresh.py` reproduces the analysis from those
snapshots, and hourly book generation updates it automatically. The current
analysis can remain pending until the candidate has actually started.

### 8. Continuation control: first twenty updates (2026-10-03)

The standard-LoRA continuation completed update 120 and its AIME25 evaluation
at 06:02 UTC. The refresh candidate has not started; these are control-only
observations, not evidence that the proposed intervention works.

| Global step | AIME25 avg@8 | AIME25 pass@8 | Correct responses | Questions with a correct response |
| --- | ---: | ---: | ---: | ---: |
| 100, fresh starting evaluation | 16.67% | 33.33% | 40 / 240 | 10 / 30 |
| 120 | 19.17% | 40.00% | 46 / 240 | 12 / 30 |

Raw response records verify thirty questions with eight samples each in both
evaluations. The observed gains are 2.50 percentage points in avg@8 and 6.67
points in pass@8; thirty questions and stochastic decoding are insufficient
to establish a stable improvement. Historical step-100 evaluation scores
are not substituted for this continuation's newly sampled starting scores.

Across updates 101–120, mean training correctness is 37.79%, mean gradient
norm is 0.04219, and mean cosine between successive effective weight updates
is 0.90199 over nineteen measured pairs. This provides a measured control
trajectory for the refresh experiment; it does not identify which part of a
cold reset caused the earlier deficit. The step-120 checkpoint contains all
eight model, optimizer and extra-state shards plus trainer and data state.
The shared historical source checkpoint remains retained.

### 9. Continuation control: step-140 evaluation (2026-10-03)

The control completed update 140 and evaluation at 07:32 UTC. AIME25 avg@8
is 17.92% (43 correct responses / 240), and pass@8 is 40.00% (12 / 30
questions). Raw records verify eight samples for each of thirty questions.
Compared with step 120, avg@8 falls 1.25 percentage points, while pass@8
is unchanged. Compared with the fresh step-100 starting evaluation, the
scores remain 1.25 and 6.67 points higher respectively. These small sampled
changes do not establish sustained improvement or deterioration.

At update 140, gradient norm is 0.02861, entropy is 0.10828, and recent
ten-update training correctness is 39.45%. The native step-140 checkpoint
contains all eight model, optimizer and extra-state shards. The proposed
refresh candidate remains queued; no intervention comparison is available.

### 10. Continuation control: step-160 evaluation (2026-10-03)

The control completed update 160 and its evaluation before training resumed
at 09:00 UTC. AIME25 avg@8 is 19.58% (47 correct responses / 240), and
pass@8 is 40.00% (12 / 30 questions). Saved response records verify thirty
distinct prompts with exactly eight samples each. Compared with step 140,
avg@8 increases 1.67 percentage points, while pass@8 remains unchanged.
Compared with the fresh step-100 starting evaluation, the gains are 2.92
and 6.67 points respectively. These fluctuations on thirty questions do
not establish sustained improvement.

The step-160 gradient norm is 0.04839. Its native checkpoint contains all
eight model, optimizer and extra-state shards plus trainer and data state.
Raw evaluation responses remain retained. The refresh candidate has not
started, so these measurements still describe only the standard-LoRA
control; they do not demonstrate that compensated refresh works.

### 11. Continuation control: step-180 evaluation (2026-10-03)

Evaluation completed before training resumed at 10:28:57 UTC. Saved records
verify thirty questions with eight samples each: 36 / 240 responses are
correct, giving 15.00% avg@8; 8 / 30 questions have a correct response,
giving 26.67% pass@8. Compared with step 160, these scores fall 4.58 and
13.33 percentage points respectively. This is an observed evaluation
decline, but one stochastic evaluation on thirty questions cannot establish
its persistence or attribute it to an optimizer mechanism.

There is an accompanying length signal. Responses stopped at the length cap
increase from 81 / 240 (33.75%) at step 160 to 116 / 240 (48.33%) at step
180. Mean evaluation length increases from 5,370 to 5,718 tokens. This is
consistent with more difficulty finishing solutions within the fixed 8,192
token budget; it does not prove that extending the cap would recover
accuracy. The matched trial's response budget remains unchanged.

Meanwhile, mean training correctness rises slightly from 40.27% over updates
141–160 to 40.96% over 161–180. Mean response length rises from 4,078 to
4,269 tokens, entropy changes from 0.1194 to 0.1176, and gradient norm from
0.03828 to 0.04188. The evaluation decline therefore does not coincide with
vanishing gradients or a decrease in this training-window correctness.
The step-180 native checkpoint contains all 26 state files. Automatic
checkpoint retention has removed the prior step-160 checkpoint; its raw
evaluation responses are preserved. The shared source checkpoint remains
retained. The refresh candidate has not started, so no method comparison
can yet be made.


### 12. Completed continuation control and refresh launch (2026-10-03)

The standard rank-one control reached global step 200 and exited with status
zero. The final evaluation completed at approximately 12:01 UTC. All 100
new updates, numbered 101–200, are present. This is a continuation from the
shared historical step-100 checkpoint, not a new run from the base model.

| Global evaluation step | Correct responses / 240 | AIME25 avg@8 | Solved questions / 30 | Pass@8 |
| --- | ---: | ---: | ---: | ---: |
| 100, fresh starting evaluation | 40 | 16.67% | 10 | 33.33% |
| 120 | 46 | 19.17% | 12 | 40.00% |
| 140 | 43 | 17.92% | 12 | 40.00% |
| 160 | 47 | 19.58% | 12 | 40.00% |
| 180 | 36 | 15.00% | 8 | 26.67% |
| 200 | 41 | 17.08% | 10 | 33.33% |

The final dump contains exactly thirty distinct questions with eight
responses each. Relative to its fresh starting evaluation, final avg@8
increases only 0.42 percentage points and pass@8 is unchanged. The observed
step-160 peak is not the endpoint: selecting it would overstate sustained
improvement. One stochastic evaluation per checkpoint on thirty questions
cannot establish a precise learning trend or statistical equivalence.

| Update window | Training correctness | Entropy | Gradient norm | Response tokens | Effective update L2 | Successive-update cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 101–120 | 37.79% | 0.1261 | 0.04219 | 4,036 | 0.06714 | 0.9020 |
| 121–140 | 39.39% | 0.1207 | 0.04452 | 4,249 | 0.06944 | 0.9043 |
| 141–160 | 40.27% | 0.1194 | 0.03828 | 4,078 | 0.06035 | 0.9020 |
| 161–180 | 40.96% | 0.1176 | 0.04188 | 4,269 | 0.06362 | 0.8951 |
| 181–200 | 39.61% | 0.1105 | 0.04382 | 4,466 | 0.06713 | 0.9016 |

Each window has twenty updates. The first cosine mean has nineteen pairs
because the historical previous-update diagnostic was unavailable at the
resume boundary. Training batches change over time; these means are
descriptive and are not repeated measurements of fixed questions.

Training correctness rises modestly and then retreats, while response
length grows and entropy falls. Weight-space update norms remain nonzero
and successive updates remain strongly aligned. These signals argue against
vanishing updates as an explanation for the control's weak final evaluation
gain. They do not establish that accumulated updates improve held-out math.
At step 200, 115 / 240 evaluation responses (47.92%) stop at the length cap,
compared with 116 at step 180 and 81 at step 160. Persistent truncation is
an accompanying signal; changing the response budget would invalidate this
matched comparison and is deferred.

The control reports zero merge cycles and mean accumulated stable rank
approximately one. The largest recorded trainer allocator peaks across all
eight ranks and all new updates are **17.955 GiB allocated** (rank 6,
step 162) and **18.828 GiB reserved** (rank 7, step 131). These cover policy
forward/backward and optimizer work, not total device memory during rollout.
Ten-second NVML records include vLLM's resident cache and all device
processes, so they answer a different memory question.

On successful completion, the queue removed the completed control's native
checkpoints (17,760,116,591 bytes) and dense policy export
(17,657,172,362 bytes). Metrics, logs, rank and memory diagnostics, and all
raw evaluation response dumps remain available. The original shared
step-100 checkpoint is retained for the candidate. Cleanup accounting is
saved in the run's `checkpoint-cleanup.json`.

The queue then launched `qwen3-4b-base-grpo-relora-refresh-r1-continue-20261003-01`.
Its launcher PID 1529014 was verified live at 12:02 UTC. Launch does not
prove successful resume or refresh: the native optimizer audit and actual
step-101 boundary diagnostics must be checked when produced. No candidate
reward or evaluation result exists yet, so there is still no evidence that
the proposed refresh matches or exceeds standard LoRA. The fixed comparison
uses the endpoint and aligned windows above; the control's plateau does not
lower the requirement to demonstrate useful candidate learning and repeat
any promising result from the base model.


### 13. First actual compensated refresh: step 101 (2026-10-03)

The candidate restored the exact shared native policy checkpoint with AdamW,
all Adam counters at 100, scheduler counter 100 and constant LR 1.5e-5.
Its fresh starting evaluation contains thirty questions with eight responses
each: 45 / 240 correct responses (18.75% avg@8), and 13 / 30 questions
solved at least once (43.33% pass@8). The control's fresh starting evaluation
was 16.67% / 33.33%. Neither policy had received a new update or refresh at
these evaluations. The difference is sampling variation, not a method gain;
paired-question analyses must retain both starting evaluations.

Step 101 completed and refreshed all 252 LoRA projections. The rollout
correctness was 50.00%, entropy 0.1261, gradient norm 0.03747 and mean
response length 3,476 tokens. These responses were sampled **before** the
optimizer update and refresh. They do not measure the intervention's effect.
The actual optimizer weight-space update L2 was 0.05901, excluding the
compensation itself. The first update has no successive-update cosine
because the source checkpoint did not contain this diagnostic history.

| Actual first-boundary diagnostic | Value |
| --- | ---: |
| A rotation | 20 degrees |
| B first-moment projection multiplier | 0.9396926 |
| Optimizer state entries retained | 903 |
| Base compensation L2 | 1.18603 |
| Relative FP32 storage-rounding L2 | 0.00003044 |
| Real response-prefix probe tokens | 1,024 |
| Mean response-prefix KL | 0.00042585 |
| Mean absolute chosen-token log-probability change | 0.006893 |
| Maximum absolute chosen-token log-probability change | 0.249574 |
| Argmax flip fraction | 0.1953% |
| Mean accumulated stable rank | approximately 1 |

The probe covers the first 128 response tokens on each of eight ranks.
Measured KL is small but nonzero, and the maximum token difference is not
negligible. This is not exact numerical policy preservation and does not
exclude larger effects later in a long response. CPU algebraic tests and
BF16 fixture checks are separate evidence from this real-model measurement.

The rank-one result immediately after refresh is expected. With B held
fixed, the compensation plus active update sum to the original `scale * B @ A`
in real arithmetic. Rotating A creates an opportunity for subsequent B
updates to introduce independent directions; it cannot instantly increase
the effective rank without changing the function. Rank growth and useful
reward progress therefore require later updates. State-entry count confirms
states were not wholesale cleared, but by itself does not prove correct
moment transport; that behavior is covered by the implementation and state
identity tests, while the native resume audit independently verifies counters.

The trial resumed rollout for step 102 after this boundary. At this point,
the evidence supports successful execution of the proposed refresh, not
successful learning or parity with standard LoRA. The first aligned reward
window and held-out comparison are still pending at step 120.


### 14. First matched learning window: updates 101–120 (2026-10-03)

The candidate completed evaluation before resuming training at 13:34:22 UTC.
The saved response dump verifies thirty questions with eight samples each:
41 / 240 responses are correct (17.08% avg@8), and 12 / 30 questions have
at least one correct sample (40.00% pass@8). Eighty-eight responses (36.67%)
stop at the length cap. The native step-120 checkpoint contains all 26
model, optimizer, extra-state, trainer and data files.

| Measurement | Standard continuation | Compensated refresh | Refresh minus standard |
| --- | ---: | ---: | ---: |
| Step-100 avg@8, before new training | 16.67% | 18.75% | +2.08 points |
| Step-120 avg@8 | 19.17% | 17.08% | −2.08 points |
| Step-100 pass@8, before new training | 33.33% | 43.33% | +10.00 points |
| Step-120 pass@8 | 40.00% | 40.00% | 0.00 points |
| Mean training correctness, 101–120 | 37.79% | 37.32% | −0.47 points |
| Mean entropy, 101–120 | 0.12611 | 0.12603 | −0.00008 |
| Mean gradient norm, 101–120 | 0.04219 | 0.04326 | +0.00107 |
| Mean response tokens, 101–120 | 4,036 | 4,099 | +63 |
| Mean effective update L2, 101–120 | 0.06714 | 0.06446 | −0.00268 |
| Mean successive-update cosine, 101–120 | 0.90199 | 0.88692 | −0.01507 |

Each training window contains twenty updates; cosine means contain nineteen
pairs. The optimizer-update diagnostic excludes compensation. Training
correctness closely tracks the control, entropy and response length are
similar, and effective update magnitude is only about 4% smaller. There is
no immediate collapse in update magnitude or training correctness in this
window. These diagnostics cannot determine whether updates are useful for
held-out math, and this single window cannot exclude a later delayed deficit.

The evaluation does **not** show improvement over the control. A paired
question bootstrap with 10,000 resamples and seed 42 gives a step-120 avg@8
difference of −2.08 percentage points, with a 95% interval of [−5.42, +1.25].
The pass@8 difference is zero, with interval [−13.33, +13.33]. Whole questions
are the sampling units, retaining their eight-response groups. These
intervals cover question sampling uncertainty, not training-seed uncertainty;
including zero does not prove equivalence.

The candidate also started with higher stochastic scores before any refresh.
Its observed avg@8 change is −1.67 points, versus +2.50 for the control;
the difference in changes is −4.17 points, with descriptive paired-question
interval [−11.25, +2.93]. Pass@8 changes are −3.33 and +6.67 points, giving
−10.00 points with interval [−26.67, +6.67]. This starting-score adjustment
is not a causal estimate and does not remove decode noise. Both raw endpoint
and starting evaluations remain visible in the comparison.

Only the first-boundary spectrum has been logged so far, where accumulated
rank was approximately one as expected from compensation. This evaluation
does not establish subsequent rank growth; the next boundary at step 141
will supply another accumulated-spectrum measurement. The next checks are
whether the training curves continue to track standard over updates 121–140,
whether the held-out result recovers, and whether rank growth accompanies
useful updates. No recipe is changed mid-run, and parity or superiority
remains unproven.


### 15. Second hourly refresh check: step 126 (2026-10-03)

At 14:03:43 UTC, the hourly watcher verified the original launcher PID
1529014 with its recorded process start time, still running with 126
completed updates. Both the snapshot export and research-page/figure build
completed successfully. No restart was needed. The latest ten updates have
39.26% training correctness; step 126 has entropy 0.12097, gradient norm
0.03664 and mean response length 4,530 tokens. These changing-batch values
are descriptive, not a new held-out result. Evaluation remains at step
120; the next evaluation is scheduled for 140 and the next refresh for 141.

Through update 123, all eight ranks contributed 184 trainer-memory records.
The observed maximum was 17.864 GiB allocated and 18.730 GiB reserved, both
at update 110 on rank 5. The completed control's corresponding maxima over
all 100 new updates were 17.955 and 18.828 GiB. These observation windows
are unequal: the candidate's interim peak does not establish memory saving.
This measurement covers training allocator memory, not total GPU usage
with the rollout engine. Final comparisons must use the full candidate run.

No new evidence yet establishes performance parity or improvement. Keep
the matched recipe and evaluate the subsequent complete windows and
held-out dumps before deciding whether to refine or replicate this method.


### 16. Main comparison scope: 100 updates from the base model

The user clarified that the established comparison budget is 100 updates
from Qwen3-4B-Base. The currently running 100-to-200 continuation is an
additional boundary diagnostic: it branches the same learned policy and
native Adam state, but cannot substitute for a base-model learning-curve
comparison. Its results must remain in the separate continuation figure.
The extra diagnostic was chosen without explaining this distinction clearly
before launch. It is not a change to the main success criterion.

Prepared profiles `configs/qwen3-4b-base-grpo-relora-refresh-r1.json` and
`configs/qwen3-4b-base-grpo-standard-r1-refresh-control.json` start from the
base model, with no resume path, and stop after 100 actual updates. Both
retain the full-layer rank-one, alpha-32, AdamW 1.5e-5 constant-LR GRPO
recipe, eight rollouts per prompt, 32 prompts per update, all eight GPUs,
8,192 response tokens, and AIME25 evaluation every twenty updates. The
candidate performs compensated 20-degree refreshes after updates 40 and 80;
the matched control disables refresh. Only those two method keys differ.
Checkpoints every twenty updates allow recovery; dense export is at 100.
No new experiment has been launched for these prepared profiles. Current
continuation evidence should inform the design before committing more GPU
compute. Historical rank-one LoRA remains the main comparison reference;
a fresh control with the same instrumentation would additionally check
implementation and sampling variation.


The user subsequently rejected the continuation as the experiment direction.
It was manually stopped at 14:38:51 UTC on 2026-10-03; the queue and
launcher descendants were terminated, and logs and measurements preserved.
It must be reported as an interrupted diagnostic, not a completed method
comparison. No replacement run was launched. The next substantive task is
reviewing the proposed algorithm against the observed first-merge learning
gap, then using the established base-model 100-update comparison.


### 17. Corrected algorithm experiment: from the base model, 100 updates

Before launch, the causal hypothesis remains specific: cold B=0 removes
`scale * B.T @ G` from A's gradient, random A changes B's gradient direction,
and clearing Adam discards accumulated optimization history. Gradual refresh
keeps B nonzero and the effective weight unchanged in real arithmetic;
therefore the same-trajectory A gradient remains unchanged. Rotating A by
20 degrees retains about 94% of its old component, introduces a new input
direction, and keeps the SGD first-order projector change bounded as derived
above. This motivates a bounded exploration of new directions while keeping
useful updates active. It is not an Adam or RL convergence guarantee.

The 20-degree angle is retained from the predeclared proposal rather than
selected using the interrupted diagnostic. The remaining approximation is
B's Adam history: unknown gradients along the new direction cannot be
reconstructed. Preserve counters and variances; multiply its first moment
by the retained cosine component. Actual optimizer update norms/alignment,
response-prefix KL and accumulated spectra will be recorded to test whether
this approximation creates an optimization delay of its own.

The main candidate is `qwen3-4b-base-grpo-relora-refresh-r1-20261003-01`.
Use the historical standard rank-one LoRA 0-100 trial as the established
reference. The candidate has the same initial Kaiming-A/zero-B LoRA, data,
GRPO recipe, optimizer, LR, eight rollouts, prompt batch and response budget.
Only its later refresh algorithm differs; checkpoint frequency is twenty
updates for recoverability. The prepared fresh standard-control profile can
check instrumentation if any pre-refresh mismatch appears; it is not being
launched as an additional experiment now.

Predeclared assessment: examine the full reward curve, especially the old
46-60 deficit; compare evaluation at 0/20/40/60/80/100 and the final endpoint,
not the best observed checkpoint; measure useful update continuity through
both boundaries and actual accumulated rank after subsequent learning.
Standard LoRA's historical final AIME25 avg@8/pass@8 is 20.0%/43.33%.
One sampled run on thirty questions cannot establish precise equivalence.
A promising result requires replication before claiming sustained parity.
Hourly process checks, ten-second NVML samples and actual training allocator
peaks remain enabled. The interrupted continuation stays in its own figure
and will not be counted as a completed main-budget trial.
