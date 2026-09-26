# Methods and measurement

UNORL studies resource-efficient on-policy adaptation. Current comparisons use
Qwen3-4B-Base, rank-1 LoRA on all linear layers (alpha 32), AdamW at 1.5e-5, one
update per rollout batch, no KL or entropy bonus, 8,192 response tokens, temperature
1.0 and top-p 1.0. Each experiment page exposes its actual configuration; older
runs differ from this profile. The current measurements use eight A100 GPUs and
are not evidence that the full process already fits a laptop.

## Metrics

- **Training correctness:** `reward/mean_positive_reward`, the fraction of sampled
  responses receiving +1. Signed mean reward is `2 × correctness − 1`; it can be
  negative without implying negative accuracy.
- **Avg@8:** correct responses divided by all eight samples for each question,
  averaged over questions. This estimates sampled pass@1 using eight draws;
  it is not greedy accuracy.
- **Pass@8:** questions with at least one correct answer among eight samples divided
  by question count. AIME25 and AIME26 each contain 30 questions. One question
  changes pass@8 by 3.33 percentage points.
- **Entropy and gradient norm:** reported directly from each trainer. Absolute
  comparisons require identical loss reduction and metric implementation. In
  particular, the archived conditional-SFT entropy should not be interpreted as
  directly comparable to the RL entropy without checking its historical code.

Plots show individual evaluation points without smoothing. Training curves show a
trailing 10-update mean with the underlying values faintly visible. Missing metrics
remain missing. There are no multi-seed confidence intervals in the retained data.
The faint traces are observations, not uncertainty bands.

## Interpreting the final REINFORCE comparison

At steps 60, 80 and 100, batch-normalized REINFORCE produced respectively 42, 31
and 43 correct responses out of 240, covering 13, 5 and 9 questions. Thus avg@8
recovered at the end while pass@8 remained below its peak. These draws alone cannot
separate sampling noise from a change in per-question capability. Different
step-0 draws also differ, but that observation alone does not prove that every
later decline is noise. Repeated stochastic evaluations would be needed.

The normalized reward is `(R − mean(R)) / std(R)`, with a population standard
deviation and a numerical floor. For binary signed rewards and success fraction
`p`, the two advantages are `sqrt((1−p)/p)` and `−sqrt(p/(1−p))`. This changes the
per-sample weighting and gradient scale; it does not by itself diagnose an observed
performance decline. Mean-centering is an action-independent baseline only in the
appropriate expectation; estimating it from the same finite batch adds estimator
considerations. No causal explanation is established by these runs.

## Data and runtime

Training uses the benchmark-cleaned parquet derived from
`eshwarprasadS/DAPO-Math-8k-Stratified`. Historical snapshots keep the selected paths
and hashes of data-audit files. No dataset contents or model weights are published
in this repository. The prepared training parquet has `prompt` (chat-message list),
`data_source`, `ability`, `reward_model` (ground-truth answer and rule style), and
`extra_info`. Evaluation also sets `env_class` to `benchmark_math`. Benchmark
prompt/answer overlap must be audited before creating a new training dataset.
The retained [training audit](data/training-audit.json) removed eight of 7,500
training examples (7,492 retained) by whitespace-normalized question substring
matching against MATH500, AMC23 and AIME25. This audit does not detect paraphrases
and does not establish AIME26 decontamination.

SkyRL is pinned at `0b286bacba2bb51dfe50186b6b5d6b1e0b5f5518`. Training uses
`unorl.grading.MathAnswerEnv` registered as `unorl_math`; evaluation uses
`unorl.benchmark_eval.BenchmarkEnv`. The shared system prompt is:

> Please reason step by step. End with Answer: \boxed{your final answer}.

The reward uses the final explicit answer. Benchmark grading additionally uses
bounded symbolic verification. The active training implementation preserves the
baseline prompt serialization. Older conditional-SFT code is retired; its original
source snapshots and raw logs remain local, with its measured results in the book.

## Provenance and scope

The book's JSON snapshots contain metric series, portable configurations, source
SHA-256 hashes, recorded exit status when available, and counts derived from valid
evaluation dumps. They do not claim missing checkpoints or incomplete logs exist.
A truncated evaluation dump is disclosed on its experiment page. Configuration-only
launches have pages with no fabricated plots. Orchestration directories are not
independent experiments. Smoke runs are excluded and listed in the cleanup manifest.
