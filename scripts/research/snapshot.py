"""Export curated local experiment metrics; no weights or response text are published."""

import hashlib
import json
from pathlib import Path

NOTES = {
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
        "Full-policy GRPO provides the original learning reference. Its learning rate and warmup differ from the later LoRA profiles, so this is not an isolated adapter ablation.",
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
        exitfile = root / "runs/logs" / f"{rid}.exit-status"
        if exitfile.exists():
            result["exit_status"] = exitfile.read_text().strip()
        (out / f"{rid}.json").write_text(json.dumps(result, indent=2) + "\n")
        registry.append({"id": rid, "title": title, "group": group})
    (out / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print("Exported", len(registry), "auditable run snapshots")


if __name__ == "__main__":
    main()
