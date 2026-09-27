# Batch-normalized REINFORCE after step 60

The earlier account emphasized evaluation noise too strongly. The retained outputs
show a second signal: a substantial increase in unfinished, truncated answers.
This is a concrete candidate mechanism to test; it does not establish causality.

## What changed

| Batch-normalized REINFORCE | Step 60 | Step 80 | Step 100 |
|---|---:|---:|---:|
| Correct responses / 240 | 42 | 31 | 43 |
| Questions solved / 30 | 13 | 5 | 9 |
| Truncated responses / 240 | 39 | 63 | 101 |
| Truncation rate | 16.3% | 26.3% | 42.1% |
| Mean evaluation response tokens | 4,335 | 4,859 | 5,601 |
| Correctness among completed responses | 20.9% | 17.5% | 30.9% |

Every truncated response in these evaluations was incorrect. The four questions
solved at step 60 but not at step 100 (dataset indices 1, 4, 18, 28) had 6/32
truncated responses at step 60 and 17/32 at step 100. The remaining nine solved
questions accumulated more correct responses. Consequently, average sample accuracy
recovered while coverage declined. This is compatible with both stochastic changes
and a shift toward reliably solving fewer questions within the generation budget.

```{figure} ../figures/truncation-diagnosis.svg
:alt: Truncation, response length and completed-response correctness across three algorithms.

All points come from valid retained AIME25 evaluation dumps. Missing or malformed
dumps are omitted. Completed-response correctness is conditioned on stopping and
must not replace overall accuracy as the success criterion.
```

The GRPO reference's truncation rate rose from 17.1% to 26.7% over steps 60–100;
vanilla REINFORCE rose from 12.1% to 35.8%. Batch-normalized REINFORCE grew fastest.
There is no accompanying gradient explosion: its mean gradient norm was about
0.054 over steps 41–60 and 0.053 over steps 81–100. Training response correctness
averaged 34.6%, 37.5%, and 38.1% over steps 41–60, 61–80, and 81–100 respectively.
Its late improvement rate flattened while average training response length grew
from 3,277 to 4,189 tokens. GRPO's final entropy is similar, so entropy alone does
not explain the difference.

## The interaction with loss masking

The original profile sets `generator.apply_overlong_filtering=true`. SkyRL zeroes
the entire loss mask for a response whose stop reason is not `stop`. Its reward
still enters the batch mean and standard deviation used to compute advantages.
Thus an unfinished incorrect rollout gets a negative advantage, but contributes
no negative policy update.

The estimator centers all N trajectory scores. If T of them are truncated failures
with advantage `a_minus < 0`, discarding their gradients leaves the unweighted sum
of retained advantages equal to `−T × a_minus > 0`. This arithmetic demonstrates
that centering no longer holds on the retained trajectories. It is not a proof
that the expected parameter gradient points toward longer answers: token lengths,
score-function directions, and importance weights also matter. The prior run did
not log its training truncation fraction, so the scale of this effect on its
training batches cannot be reconstructed from evaluation data.

Filtering is a deliberate design choice, not a universal implementation bug.
[DAPO](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf) reported that masking
truncated samples helped its training by avoiding penalties on potentially sound
but unfinished reasoning; it also studied soft length penalties. Our single-rollout,
rank-1 setting and fixed 8k answer budget differ. Reversing the filter is an ablation
motivated by our measurements, not a claim that DAPO's finding is wrong.

## Controlled trial: retain truncated failures

The new profile is `qwen3-4b-base-reinforce-batchnorm-retain-truncated-r1.json`.
Its only configuration change from the completed AdamW baseline is:

```yaml
generator.apply_overlong_filtering: false
```

The verifier and reward mapping remain +1/−1; a truncated response with a correct
explicit answer is not forcibly relabeled. The responses keep their normal token
masks, so unsuccessful unfinished trajectories can receive negative updates.
The model starts from the same Qwen3-4B-Base checkpoint, with rank-1 LoRA, AdamW
1.5e-5, batch 256 × one rollout, 8,192 response tokens, no KL/entropy bonus, no
warmup, 100 updates, and all eight A100s. AIME25 uses avg@8 and pass@8 every 20 steps.
This is a fresh run, not a continuation from the retrospectively best checkpoint.

Changing the retained tokens also changes the token-mean denominator; this is
part of the filtering ablation and may reduce effective update magnitude. The
experiment cannot separately identify the effect of restoring negative samples
and the accompanying normalization-scale change.

## Diagnostics and success criteria

The new trainer logs truncation fraction, retained response/token fraction,
masked incorrect fraction, outcome standard deviation, advantage mean over active
and completed trajectories, token-weighted advantage mean, and the fraction of
negative advantage mass carried by truncated responses. These describe masks before
SkyRL's final loss reduction and importance correction and do not modify rewards.

First verify that failed truncated responses actually receive negative gradients
and updates remain finite. For learning, compare the last 40 updates with the old
run, using response correctness together with AIME25 avg@8 and pass@8. Lower
truncation or shorter responses alone do not count as success. Improvements should
persist over multiple evaluation checkpoints; one peak on 30 questions cannot
establish stable improvement. If the trial suppresses useful reasoning or degrades
accuracy, retain the result as a failed ablation before changing another factor.

The [analysis snapshot](../data/truncation-analysis.json) records per-question
counts, source hashes, missing dumps, and training-window means. Regenerate it
locally with `python scripts/research/analyze_truncation.py`; the book renderer
uses the committed snapshot without requiring raw run files.

## Outcome of the 100-step trial

The trial completed successfully (exit status 0). All training responses remained
trainable (`rollout/trainable_response_fraction=1`) and no incorrect response was
masked at the recorded checkpoints. Training truncation reached 11.3% at steps 80
and 100. At step 100, truncated failures carried 34.0% of the negative advantage
mass before token-mean reduction. The active response mean advantage was essentially
zero, as expected when all 256 responses enter batch centering.

| AIME25 | Step 60 | Step 80 | Step 100 |
|---|---:|---:|---:|
| Retain truncated: avg@8 | 15.8% | 16.25% | 14.6% |
| Retain truncated: pass@8 | 23.3% | 30.0% | 30.0% |
| Retain truncated: truncated / 240 | 41 | 36 | 34 |
| Original filtering: avg@8 | 17.5% | 12.9% | 17.9% |
| Original filtering: pass@8 | 43.3% | 16.7% | 30.0% |
| Original filtering: truncated / 240 | 39 | 63 | 101 |

The new policy generated far fewer unfinished evaluation answers at step 100
(34/240 versus 101/240), so the changed gradient reached the behavior it targeted.
It did not produce a higher final AIME25 accuracy: the last avg@8 was 14.6%
versus 17.9%, and both runs solved 9 of 30 questions at least once in eight
samples. The new run's pass@8 stopped falling after step 60, but its avg@8
slipped from step 80 to 100. A single 30-question evaluation cannot establish
a reliable small difference. This trial does **not** demonstrate stable learning
improvement, despite reducing truncation.

The step-0 and step-20 raw evaluation dumps in this trial are malformed; their
aggregate metrics remain in the trainer log. Valid response-level counts support
the step-40, 60, 80, and 100 truncation figures above. A useful next ablation is
to retain the failed truncated responses with a smaller or tapered negative
weight, while keeping the loss denominator fixed. That would test whether the
full negative update over-suppresses potentially useful reasoning.
