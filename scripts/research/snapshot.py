"""Export curated local experiment metrics; no weights or response text are published."""

import hashlib
import json
from pathlib import Path

NOTES = {
    "qwen3-4b-base-grpo-relora-r1-warmup0-20261002-01": (
        "GRPO · standard rank-1 ReLoRA, constant-LR resets",
        "ablation",
        "Completed all 100 steps with exit status 0. Matched standard rank-one Kaiming LoRA / alpha 32 GRPO settings and complete AdamW state clears at merges 40/80; only post-merge restart ramp differs from the paired trial (disabled here). Successive 20-step training-correctness means: 11.895%, 22.051%, 32.793%, 37.324%, 37.031%. Final AIME25 avg@8 / pass@8: 17.083% / 30.0%, versus 15.833% / 40.0% for the ramp and 20.0% / 43.333% for standard LoRA. Step-80 evaluation peaked at 18.333% / 50.0%; final results fluctuate on 30 questions. Boundary response-prefix KL was 0.0005605 / 0.0005378 at steps 40/80, from 1024 total probe tokens per boundary. Final per-layer mean accumulated stable rank was 2.160, with 53.25% mean update energy outside the leading direction. Peak training allocated/reserved memory across eight ranks was 18.60/19.47 GiB, including scheduled merge diagnostics. Neither pair shows an immediate post-merge reward collapse or a clear improvement over the standard LoRA reference. Checkpoint/model weights and factor cache removed after completion. Cleanup also erroneously removed raw evaluation response dumps under exports; aggregate metrics, evaluation.jsonl, logs, memory traces and rank diagnostics survive. The cleanup rule now targets policy/critic weight directories and preserves benchmark dumps; a regression test verifies this. Response-level regrading is unavailable for this run.",
    ),
    "qwen3-4b-base-grpo-relora-r1-warmup5-20261002-01": (
        "GRPO · standard rank-1 ReLoRA, five-update restart ramp",
        "ablation",
        "Completed all 100 steps with exit status 0 after fixing the trajectory-prefix diagnostic. Standard Kaiming rank-one LoRA / alpha 32, native AdamW LR 1.5e-5, eight rollouts, 32 prompts/update, 8192 response tokens; merge/reset at 40/80, then five-update LR multipliers 0, .25, .5, .75, 1; no initial warmup. Successive 20-step training-correctness means: 12.656%, 23.281%, 32.695%, 36.348%, 36.309%. Final AIME25 avg@8 / pass@8: 15.833% / 40.0%, versus 17.083% / 30.0% for constant-LR resets and 20.0% / 43.333% for standard LoRA. There is no clear benefit from this ramp in one sampled run per setting. Boundary response-prefix KL was 0.0005664 / 0.0005645 at steps 40/80, using 1024 real response-prefix tokens per boundary. Final mean accumulated stable rank was 1.887, with 46.48% mean update energy outside the leading direction. Rank growth is measurable but did not establish better learning. Peak training allocated/reserved memory across eight ranks was 17.96/18.95 GiB. Neither merge produced an immediate reward collapse. Earlier failed attempt is separately retained and never reached a merge. Checkpoint/model weights and factor cache removed after completion. Cleanup also erroneously removed raw evaluation response dumps under exports; aggregate metrics, evaluation.jsonl, logs, memory traces and rank diagnostics survive. Cleanup was fixed and regression-tested. Response-level regrading is unavailable for this run.",
    ),
    "qwen3-4b-base-grpo-relora-r1-warmup5-20261001-01": (
        "GRPO · ReLoRA restart ramp, failed probe attempt",
        "ablation",
        "Stopped with exit status 1 during step 40, before the first merge or optimizer update at that step; 39 updates completed. A diagnostic indexing bug treated the start of the common response slice as the prompt boundary. With native SkyRL whole-sequence left padding, short responses can have their prompt inside that slice, so the probe incorrectly raised an empty-prompt error. Native training itself was unaffected. Correctness over steps 1–10, 11–20, 21–30 and 31–39 was 10.4%, 17.1%, 21.1%, 27.7%; standard LoRA reference over the same ranges was 9.65%, 14.84%, 20.82%, 25.95%. AIME25 at step 20 was 6.25% avg@8 / 26.67% pass@8, versus 4.17% / 23.33% in the reference. These are pre-merge single-run results and do not establish a ReLoRA benefit. No boundary/rank results were produced. The control did not start. Fixed and restarted from the base model on October 2; original logs and evaluations retained.",
    ),
    "qwen3-4b-base-grpo-loft-simple-r1-20261001-01": (
        "GRPO · full-layer rank-1 LoFT-simple",
        "ablation",
        "Launched a 100-step GRPO comparison using the authors' LoFT-simple optimizer geometry: alternate B/A updates starting with B; rescale gradients and transport first moments; update moments for both factors every step; no second-moment transport or merge. Retain the reference's Kaiming initialization, LR 1.5e-5, beta values, batch, eight rollouts and response budget. Method-required differences include alpha 1 and Adam epsilon 1e-4, versus alpha 32 / epsilon 1e-8 in the reference. Rank-one specialization avoids dense weight-shaped calibration tensors; saved previous factors and alternation phase are part of the checkpoint. Eight-step CPU parity with the pinned authors' implementation and clipping agreed to within 1.5e-8 in parameters. Actual eight-GPU FSDP2 checks verified frozen backbone, A/B optimizer state, adapter export and exact next-update reproduction after resume. Calibrated active gradient norms are not directly comparable to ordinary LoRA norms. Completed 100 steps successfully. Successive 20-step training-correctness means were 8.20%, 9.24%, 8.46%, 9.22%, and 8.89%; this run did not establish meaningful learning. Final AIME25 avg@8 / pass@8 was 3.33% / 20.0%, versus 20.0% / 43.3% for the standard LoRA reference. Peak training allocated / reserved memory was 17.75 / 18.60 GiB. Checkpoints and export weights were removed after completion; measurements retained. Results do not establish that the authors' method generally fails.",
    ),
    "qwen3-4b-base-grpo-nora-merge-r1-20260930-01": (
        "GRPO · full NoRA-init with merge/reset",
        "ablation",
        "Completed all 100 steps successfully in 6h43m of training-loop time. Merges at steps 40 and 80 accumulated each learned adapter, created fresh normalized sign A / zero B, and cleared adapter AdamW history while preserving constant LR/scheduler. Backbone synchronization succeeded at startup and both merges. Training correctness over successive 20-step windows was 18.4%, 34.4%, 39.2%, 39.6%, and 37.2%: this run did not sustain renewed improvement after resets. AIME25 avg@8 peaked at 20.0% at step 80, then finished at 15.0% / 30.0% avg@8 / pass@8, versus 17.9% / 36.7% without merges and 20.0% / 43.3% for standard LoRA. These are sampled single-run results on 30 questions. Merge probe max logit differences were 0.25 and 0.3125, with RMS differences 0.0506 and 0.0556; merging was not bitwise function-preserving under mixed-precision forward computation. Peak training allocator usage was 18.00 GiB allocated / 18.96 GiB reserved, including merge work. This explicit Adam-reset baseline does not isolate rank growth from optimizer discontinuity or establish that merging generally fails.",
    ),
    "qwen3-4b-base-grpo-lorafa-r1-20260930-01": (
        "GRPO · full-layer rank-1 LoRA-FA",
        "ablation",
        "Completed 100 steps successfully. Freeze A and apply the regularized inverse-Gram correction to accumulated B gradients before clipping and native AdamW. The full-layer rank-one reference's Kaiming initialization, alpha 32, LR 1.5e-5, eight rollouts, batch size and response budget are retained. LoRA-FA leads the sampled avg@8 comparison at step 80 (17.9% versus standard LoRA's 13.8%), but at step 100 reaches 17.5% / 33.3% avg@8 / pass@8 versus 20.0% / 43.3%. This demonstrates substantial learning with frozen A, not a statistically established ranking or superiority to the reference. The reported gradient norm is after inverse-Gram correction and is not directly comparable to uncorrected reference norms. The historical full-parameter GRPO baseline reaches 17.9% / 36.7%, but differs in LR, warmup, advantage normalization, clipping and importance correction. No matched reference allocator measurement exists yet to quantify memory savings.",
    ),
    "qwen3-4b-base-grpo-lora-r1-nora-init-last18-20260929-01": (
        "GRPO · rank-1 NoRA-init in the final 18 layers",
        "ablation",
        "Completed 100 steps successfully. Relative to full-layer NoRA-init, only adapter placement changes: layers 18–35. Training correctness increases from 11.9% in steps 1–20 to 37.3% in steps 81–100. Final AIME25 avg@8 / pass@8 is 15.8% / 30.0%. The half-layer trial does not reproduce the full-layer NoRA-init run's early acceleration. Comparisons with standard initialization also change alpha from 32 to 1. This single run does not isolate the cause of the later plateau or establish a reliable ranking on 30 evaluation questions. Sampled training device-memory peak is 23.24 GiB; the ten-second NVML sampler can miss brief peaks and includes all resident processes.",
    ),
    "qwen3-4b-base-grpo-lora-r1-nora-init-20260928-01": (
        "GRPO · rank-1 NoRA-init",
        "ablation",
        "Completed 100 steps successfully. The main positive signal is faster early learning: training correctness averages 18.4% versus 12.2% in steps 1–20, and 33.8% versus 23.7% in steps 21–40. AIME25 avg@8 at step 20 is 9.2% versus 4.2% for standard rank-1 LoRA. Training correctness later levels off near 38–39%; this does not establish that NoRA causes the plateau. Final avg@8 / pass@8 is 17.9% / 36.7%, versus the reference's 20.0% / 43.3%. The trial changes initialization and alpha together (normalized A with alpha 1 versus Kaiming A with alpha 32), so the acceleration cannot be attributed to initialization alone. The next hypothesis is that periodic merge/reset may retain the faster learning while expanding the accumulated update beyond one fixed rank-one adapter. No merge/reset was used in this trial, and that hypothesis remains untested under this recipe.",
    ),
    "qwen3-4b-base-grpo-lora-r1-last18-20260928-01": (
        "GRPO · rank-1 LoRA in the final 18 layers",
        "ablation",
        "Completed 100-step controlled comparison with full-layer rank-1 GRPO. The same model, data, optimizer, learning rate, rollout count and response budget are used; adapters are excluded from layers 0–17. The actual synced adapter contains 1,032,192 parameters in layers 18–35, versus 2,064,384 in the reference. Learning starts more slowly, but final-window training correctness nearly catches up: 36.4% versus 37.5% over steps 81–100. Final AIME25 avg@8 is 17.9% versus 20.0%, and pass@8 is 33.3% versus 43.3% (10 versus 13 questions solved). This demonstrates learning with half the adapter parameters, but a single run on 30 questions does not establish equivalence to the reference. Peak training memory savings remain unmeasured. Failed startup attempts produced no learning results and are excluded.",
    ),
    "unorl-batchnorm-retain-truncated-20260926-01": (
        "Retain truncated failures · AdamW",
        "ablation",
        "Completed 100-step controlled follow-up. Only overlong filtering changes from the batch-normalized AdamW profile. Training masks confirm truncated failures receive updates. AIME25 truncation fell substantially by step 100, but final avg@8 was 14.6% versus 17.9% with filtering; both had 30.0% pass@8. Steps 60–100 were steadier in pass@8, but accuracy did not show stable improvement. See the batch-normalized REINFORCE diagnosis.",
    ),
    "qwen3-4b-base-grpo-20260916-01": (
        "Full-parameter GRPO",
        "reference",
        "Full-policy GRPO provides the original learning reference: AIME25 avg@8 / pass@8 is 17.9% / 36.7% at step 100. It uses LR 1e-6 and five warmup steps; the later LoRA profiles use LR 1.5e-5 without warmup. Advantage standard-deviation normalization, policy clipping and importance correction also differ. This historical result must remain visible in comparisons, but it is not an isolated adapter ablation.",
    ),
    "qwen3-4b-base-grpo-lora-r1-blog-20260923-01": (
        "Rank-1 LoRA GRPO",
        "primary",
        "Rank-1 LoRA improves over initialization with the adjusted learning rate. It uses 32 prompts × 8 responses per update; the single-rollout experiments use 256 prompts × 1 response. Response count is matched, but prompt exposure, token count and wall time are not.",
    ),
    "qwen3-4b-base-reinforce-adamw-r1-b256-20260925-01": (
        "Vanilla REINFORCE · AdamW",
        "primary",
        "The corrected AdamW run learns: training correctness and sampled AIME accuracy increase. Its final AIME25 results are below the rank-1 GRPO reference. Earlier SGD failures do not establish that single-rollout REINFORCE fails.",
    ),
    "qwen3-4b-base-reinforce-batchnorm-adamw-r1-b256-20260925-01": (
        "Batch-normalized REINFORCE · AdamW",
        "primary",
        "Learning improves rapidly, then training correctness levels off. Final AIME25 avg@8 is slightly higher than at step 60, while pass@8 falls: correct responses cover fewer questions. Sampling variation and changes in per-question competence cannot be separated from a single evaluation draw. The small final advantage over vanilla is not a statistically established ranking.",
    ),
    "qwen3-4b-base-conditional-20260916-01": (
        "Archived · conditional online SFT",
        "historical",
        "Both correct and incorrect self-generated responses receive positive cross-entropy updates under different system instructions. Evaluation improves temporarily and later deteriorates. This records the retired method; its code is no longer part of the active UNORL package. The historical entropy implementation/reduction must be checked before absolute comparisons to RL entropy.",
    ),
    "qwen3-4b-base-positive-only-20260918-01": (
        "Archived · positive-only online SFT",
        "historical",
        "Only verified correct self-generated responses are trained. This run demonstrates learning with rejection sampling. Training correctness and prompt-level training pass@8 are distinct quantities; compare response correctness with held-out avg@8, not training pass@8 with held-out pass@1.",
    ),
    "qwen3-4b-base-ppo-lora-r1-valuewarmup-20260924-01": (
        "PPO · value-model warmup",
        "historical",
        "The run was stopped after early degradation. Five critic-only warmup steps precede actor updates; critic and policy settings differ from REINFORCE. This is evidence about this PPO configuration, not a general failure of PPO.",
    ),
    "qwen3-4b-instruct-grpo-20260915-01": (
        "Archived · Instruct GRPO pilot",
        "historical",
        "This short pilot uses an instruction-tuned model and a 16k response budget. Its initial accuracy is much higher than Base. Keep it outside Base-model algorithm comparisons; no post-training AIME evaluation is retained in this run.",
    ),
    "qwen3-4b-base-reinforce-batchnorm-r1-b256-20260925-02": (
        "Batch-normalized REINFORCE · SGD",
        "confounded",
        "This earlier experiment used the custom stateless SGD worker, not the later AdamW path. Its weak result cannot isolate the reward-normalization algorithm. Overlong filtering and inference correction also differ.",
    ),
    "qwen3-4b-base-reinforce-batchmean-b256-20260924-01": (
        "Mean-centered REINFORCE · SGD",
        "confounded",
        "Batch-mean centering without standard-deviation scaling did not show a convincing gain before stopping. The SGD optimizer confounds comparison with the successful AdamW profiles.",
    ),
    "qwen3-4b-base-reinforce-tokenmean-b256-20260924-01": (
        "Token-mean REINFORCE · SGD",
        "confounded",
        "Token-mean reduction removed the earlier loss-scale problem. The custom worker still used SGD, so this stopped run is not an AdamW algorithm comparison.",
    ),
    "qwen3-4b-base-reinforce-lora-r1-blog-20260923-03": (
        "Early REINFORCE · loss-scale issue",
        "confounded",
        "The sequence-summed loss and stateless SGD differ from the corrected experiment. Large logged gradient norms make it unsuitable for drawing conclusions about the later token-mean AdamW method.",
    ),
    "qwen3-4b-base-reinforce-relora-gpu0-b16-m100-300-20260922-01": (
        "Single-GPU ReLoRA exploration",
        "confounded",
        "This combines rank-1 LoRA, periodic merge/reset, single-rollout REINFORCE, SGD and a batch of 16. It is an exploratory memory-oriented run with multiple simultaneous changes. It cannot identify which change caused its weak learning trend.",
    ),
    "qwen3-4b-base-grpo-lora-r1-gpu0-20260923-01": (
        "Single-GPU LoRA launch record",
        "incomplete",
        "Configuration is retained but there are no metrics. No learning or evaluation conclusion is possible.",
    ),
    "qwen3-4b-base-reinforce-batchnorm-r1-b256-20260925-01": (
        "Interrupted batch-normalized launch",
        "incomplete",
        "Configuration is retained but no metrics survived. Treat this as an incomplete launch, not a zero-accuracy result.",
    ),
}


def portable(value):
    """Remove machine-specific roots from asset paths in public snapshots."""
    if isinstance(value, str) and Path(value).is_absolute():
        parts = Path(value).parts
        for marker in ("models", "data", "runs"):
            if marker in parts:
                return str(Path(*parts[parts.index(marker) :]))
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    return value


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "research/data"
    out.mkdir(parents=True, exist_ok=True)
    registry = []
    for rid, (title, group, note) in NOTES.items():
        p = root / "runs" / rid
        cfg = json.loads((p / "config.json").read_text())
        rows = []
        hashes = {}
        source_bytes = {}
        for name in ["metrics.jsonl", "config.json", "training-data-audit.json"]:
            if (p / name).exists():
                source_bytes[name] = (p / name).read_bytes()
                hashes[name] = hashlib.sha256(source_bytes[name]).hexdigest()
        if "metrics.jsonl" in source_bytes:
            by_step = {}
            for line in source_bytes["metrics.jsonl"].decode().splitlines():
                r = json.loads(line)
                by_step.setdefault(r["step"], {}).update(r["metrics"])
            rows = [{"step": k, "metrics": v} for k, v in sorted(by_step.items())]

        optimizer = "SGD" if "trainer.low_resource.lora_rank" in cfg else "AdamW (SkyRL default)"
        if cfg.get("trainer.policy.model.lora.init_method") == "loft_simple":
            optimizer = "LoFTSimpleAdamW (Adam-family)"
        summary = []
        omissions = []
        for f in sorted(p.glob("exports/*/dumped_evals/global_step_*_evals/*.jsonl")):
            if f.name == "aggregated_results.jsonl":
                continue
            try:
                evalrows = [json.loads(line) for line in f.read_text().splitlines()]
            except json.JSONDecodeError:
                omissions.append(str(f.relative_to(p)))
                continue
            qs = {}
            for r in evalrows:
                qs.setdefault(r["input_prompt"], []).append(r["score"] > 0)
            summary.append(
                {
                    "step": int(f.parent.name.split("_")[2]),
                    "benchmark": f.stem,
                    "responses": len(evalrows),
                    "questions": len(qs),
                    "correct_responses": sum(sum(v) for v in qs.values()),
                    "solved_questions": sum(any(v) for v in qs.values()),
                }
            )
        result = {
            "schema_version": 1,
            "run_id": rid,
            "title": title,
            "group": group,
            "analysis": note,
            "optimizer": optimizer,
            "config": portable(cfg),
            "metrics": rows,
            "evaluation_counts": summary,
            "unreadable_eval_dumps": omissions,
            "provenance": {
                "source": f"runs/{rid}",
                "sha256": hashes,
                "skyrl_commit": (p / "skyrl-commit.txt").read_text().strip()
                if (p / "skyrl-commit.txt").exists()
                else None,
            },
        }
        for name in ("initial-adapter-audit.json", "comparison-audit.json"):
            if (p / name).exists():
                raw = (p / name).read_bytes()
                result.setdefault("audits", {})[name] = portable(json.loads(raw))
                hashes[name] = hashlib.sha256(raw).hexdigest()
        for name, key in (
            ("merge-audit.jsonl", "merge_audits"),
            ("rank-diagnostics.jsonl", "rank_diagnostics"),
        ):
            if (p / name).exists():
                raw = (p / name).read_bytes()
                result[key] = [portable(json.loads(line)) for line in raw.decode().splitlines()]
                hashes[name] = hashlib.sha256(raw).hexdigest()
        exitfile = root / "runs/logs" / f"{rid}.exit-status"
        if exitfile.exists():
            result["exit_status"] = exitfile.read_text().strip()
        (out / f"{rid}.json").write_text(json.dumps(result, indent=2) + "\n")
        registry.append({"id": rid, "title": title, "group": group})
    (out / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print("Exported", len(registry), "auditable run snapshots")


if __name__ == "__main__":
    main()
