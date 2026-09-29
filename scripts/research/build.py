"""Render the UNORL research book from portable, versioned metric snapshots."""

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "research"
COLORS = ["#2196F3", "#009688", "#FF9800", "#9C27B0", "#F44336", "#607D8B"]
GROUPS = {
    "ablation": "Controlled follow-up experiments",
    "primary": "Current algorithm comparison",
    "reference": "Full-parameter reference",
    "historical": "Earlier research",
    "confounded": "Optimizer and loss confounds",
    "incomplete": "Incomplete launch records",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B0BEC5",
        "axes.labelcolor": "#37474F",
        "text.color": "#263238",
        "xtick.color": "#546E7A",
        "ytick.color": "#546E7A",
        "axes.grid": True,
        "grid.color": "#ECEFF1",
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.hashsalt": "unorl",
        "svg.fonttype": "none",
    }
)


def series(run, key):
    pairs = [
        (r["step"], r["metrics"][key])
        for r in run["metrics"]
        if isinstance(r["metrics"].get(key), (int, float)) and math.isfinite(r["metrics"][key])
    ]
    return ([p[0] for p in pairs], [p[1] for p in pairs])


def evaluation(run, benchmark="aime25"):
    rows = []
    for r in run["metrics"]:
        m = r["metrics"]
        prefix = f"eval/{benchmark}/"
        avg = m.get(prefix + "mean_positive_reward", m.get(prefix + "avg@8"))
        pas = m.get(prefix + "pass_at_8")
        if avg is not None or pas is not None:
            rows.append((r["step"], avg, pas))
    return rows


def pct(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def status(run):
    if not run["metrics"]:
        return "No retained metrics"
    last = run["metrics"][-1]["step"]
    planned = run["config"].get("trainer.max_training_steps")
    return f"{last} steps logged" + (f" / {planned} planned" if planned else "")


def line(ax, run, key, color, label=None, smooth=False):
    x, y = series(run, key)
    if not x:
        return False
    if smooth:
        ax.plot(x, y, color=color, alpha=0.16, lw=1)
        y = [statistics.mean(y[max(0, i - 9) : i + 1]) for i in range(len(y))]
    ax.plot(
        x,
        y,
        color=color,
        lw=2,
        label=label,
        marker=None if smooth else "o",
        markersize=3 if not smooth else 0,
    )
    return True


def save(fig, name):
    fig.savefig(BOOK / "figures" / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(BOOK / "figures" / f"{name}.png", bbox_inches="tight", dpi=150)
    svg = BOOK / "figures" / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def run_plot(run):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    specs = [
        ("reward/mean_positive_reward", "Training response correctness", True, True),
        ("eval/aime25/mean_positive_reward", "AIME25 · avg@8", False, True),
        ("eval/aime25/pass_at_8", "AIME25 · pass@8", False, True),
        ("policy/policy_entropy", "Logged policy entropy", True, False),
        ("policy/grad_norm", "Policy gradient norm", True, False),
        ("generate/avg_num_tokens", "Mean generated response length", True, False),
    ]
    for ax, (key, title, smooth, percent), color in zip(axes.flat, specs, COLORS):
        exists = line(ax, run, key, color, smooth=smooth)
        ax.set(title=title, xlabel="Training step")
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if not exists:
            ax.text(0.5, 0.5, "Not recorded", ha="center", transform=ax.transAxes)
    fig.suptitle(run["title"], fontsize=16, fontweight="bold")
    save(fig, run["run_id"])


def comparison(runs, groups, name, title):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    selected = [r for r in runs if r["group"] in groups and r["metrics"]]
    for run, color in zip(selected, COLORS):
        label = run["title"].replace("Archived · ", "")
        for ax, key, smooth in zip(
            axes,
            [
                "reward/mean_positive_reward",
                "eval/aime25/mean_positive_reward",
                "eval/aime25/pass_at_8",
            ],
            [True, False, False],
        ):
            line(ax, run, key, color, label, smooth)
    for ax, title_ax in zip(
        axes, ["Training response correctness", "AIME25 avg@8", "AIME25 pass@8"]
    ):
        ax.set(title=title_ax, xlabel="Training step")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    save(fig, name)


def figure(path, caption):
    return f"```{{figure}} {path}\n:alt: {caption}\n\n{caption}\n```\n"


def experiment_page(run):
    cfg = run["config"]
    rid = run["run_id"]
    rows = evaluation(run)
    n = cfg.get("generator.n_samples_per_prompt", 1)
    batch = cfg.get("trainer.train_batch_size")
    fields = {
        "Run ID": f"`{rid}`",
        "Record": status(run),
        "Group": GROUPS[run["group"]],
        "Optimizer": run["optimizer"],
        "Model": cfg.get("trainer.policy.model.path", "not recorded"),
        "Training data": ", ".join(cfg.get("data.train_data", [])),
        "Advantage estimator": cfg.get("trainer.algorithm.advantage_estimator", "not recorded"),
        "Policy loss": cfg.get("trainer.algorithm.policy_loss_type", "not recorded"),
        "Loss reduction": cfg.get("trainer.algorithm.loss_reduction", "not recorded"),
        "LoRA rank": cfg.get(
            "trainer.low_resource.lora_rank", cfg.get("trainer.policy.model.lora.rank", 0)
        ),
        "LoRA alpha": cfg.get(
            "trainer.low_resource.lora_alpha", cfg.get("trainer.policy.model.lora.alpha", "—")
        ),
        "LoRA initialization": cfg.get(
            "trainer.policy.model.lora.init_method", "kaiming (default)"
        ),
        "Learning rate": cfg.get("trainer.policy.optimizer_config.lr", "not recorded"),
        "Warmup steps": cfg.get("trainer.policy.optimizer_config.num_warmup_steps", "not recorded"),
        "Prompts × responses": f"{batch} × {n} = {batch * n if batch else 'unknown'} responses/update",
        "Response limit": cfg.get("generator.sampling_params.max_generate_length", "not recorded"),
        "Evaluation samples/question": cfg.get(
            "generator.eval_n_samples_per_prompt", "not recorded"
        ),
        "GPUs (policy)": cfg.get("trainer.placement.policy_num_gpus_per_node", "not recorded"),
        "KL loss / reward": f"{cfg.get('trainer.algorithm.use_kl_loss')} / {cfg.get('trainer.algorithm.use_kl_in_reward')}",
        "Exit status": run.get("exit_status", "not retained"),
    }
    text = f'---\ntitle: "{run["title"]}"\n---\n\n# {run["title"]}\n\n{run["analysis"]}\n\n'
    text += "## Configuration and provenance\n\n| Setting | Value |\n|---|---|\n"
    text += "".join(f"| {k} | {v} |\n" for k, v in fields.items())
    text += f"\n[Download the metric/configuration snapshot](../data/{rid}.json). "
    text += "The snapshot includes hashes of the original local source files and the recorded SkyRL revision. Machine-specific root paths are made relative; original run identifiers remain unchanged.\n\n"
    text += "## Learning curves\n\n"
    if run["metrics"]:
        text += figure(
            f"../figures/{rid}.svg",
            "Evaluation points are unsmoothed. Training curves show raw values faintly and a trailing 10-update mean. Missing metrics are labeled explicitly.",
        )
    else:
        text += "No metric records survived, so no curve or score is fabricated.\n"
    text += "\n## Evaluation results\n\n"
    for benchmark in ["aime25", "aime26", "amc23", "math500"]:
        ev = evaluation(run, benchmark)
        if not ev:
            continue
        text += (
            f"### {benchmark.upper()}\n\n| Step | Sample accuracy | Pass@8 |\n|---:|---:|---:|\n"
        )
        text += "".join(f"| {step} | {pct(avg)} | {pct(pas)} |\n" for step, avg, pas in ev) + "\n"
    if not rows:
        text += "No AIME25 checkpoint evaluation is retained.\n"
    if run["unreadable_eval_dumps"]:
        text += "\nSome raw evaluation dumps were truncated or malformed; aggregated logged metrics above are retained, but per-question counts for these files are unavailable:\n\n"
        text += "".join(f"- `{p}`\n" for p in run["unreadable_eval_dumps"])
    if run["evaluation_counts"]:
        text += "\n## Evaluation sample counts\n\n| Benchmark | Step | Correct responses | Questions solved ≥1 time |\n|---|---:|---:|---:|\n"
        for e in sorted(run["evaluation_counts"], key=lambda e: (e["benchmark"], e["step"])):
            text += f"| {e['benchmark']} | {e['step']} | {e['correct_responses']}/{e['responses']} | {e['solved_questions']}/{e['questions']} |\n"
    text += "\n## Quantitative observations\n\n"
    train_steps, train_values = series(run, "reward/mean_positive_reward")
    if train_values:
        first_count = min(10, len(train_values))
        text += (
            f"Training response correctness averaged **{pct(statistics.mean(train_values[:first_count]))}** "
            f"over the first {first_count} logged updates and "
            f"**{pct(statistics.mean(train_values[-first_count:]))}** over the last {first_count}. "
            "These are different on-policy training batches, so this trend is not a fixed-test comparison.\n\n"
        )
    for key, label in [
        ("policy/policy_entropy", "Logged entropy"),
        ("policy/grad_norm", "Policy gradient norm"),
        ("generate/avg_num_tokens", "Mean generated response tokens"),
    ]:
        steps, values = series(run, key)
        if values:
            text += f"- {label}: {values[0]:.4g} at step {steps[0]} → {values[-1]:.4g} at step {steps[-1]}.\n"
    measured = [(step, avg, pas) for step, avg, pas in rows if pas is not None]
    if measured:
        best = max(measured, key=lambda row: row[2])
        last = measured[-1]
        text += (
            f"\nBest recorded AIME25 pass@8: **{pct(best[2])} at step {best[0]}**. "
            f"Last recorded: **{pct(last[2])} at step {last[0]}**. "
            "Selecting the peak after observing all checkpoints is optimistic; use the final result for an endpoint comparison.\n"
        )
    text += "\n## Interpretation limits\n\nOne retained run is not a multi-seed study. AIME contains only 30 questions per year; avg@8 measures sampled single-response accuracy, while pass@8 measures question coverage. A peak checkpoint is not the final result. Response-count matching does not equal token-compute matching. See [measurement conventions](../methods.md).\n"
    if cfg.get("trainer.policy.model.lora.init_method") == "nora_init":
        text += nora_observations(run)
    (BOOK / "experiments" / f"{rid}.md").write_text(text)


def nora_observations(run):
    """Keep the NoRA interpretation and paired measurements reproducible."""
    reference_id = "qwen3-4b-base-grpo-lora-r1-blog-20260923-01"
    reference = json.loads((BOOK / "data" / f"{reference_id}.json").read_text())
    text = "\n## Faster early learning: comparison with standard LoRA\n\n"
    text += f"Reference: [full-layer rank-1 GRPO]({reference_id}.md). The model, data, prompt format, AdamW LR, batch, rollout count and response cap match. Initialization and alpha change together; this is a method-and-scaling comparison.\n\n"
    text += figure(
        "../figures/comparison-nora.svg",
        "NoRA-init versus standard full-layer LoRA. Same number of responses per update; longer responses mean token compute is not matched.",
    )
    text += "\n### Training correctness by 20-step window\n\n| Steps | NoRA-init | Standard LoRA | Difference |\n|---|---:|---:|---:|\n"
    for start in range(1, 101, 20):
        means = []
        for current in (run, reference):
            values = [
                row["metrics"]["reward/mean_positive_reward"]
                for row in current["metrics"]
                if start <= row["step"] < start + 20
                and "reward/mean_positive_reward" in row["metrics"]
            ]
            means.append(statistics.mean(values))
        text += f"| {start}–{start + 19} | {pct(means[0])} | {pct(means[1])} | {(means[0] - means[1]) * 100:+.1f} pp |\n"
    text += "\n### AIME25 checkpoint comparison\n\n| Step | NoRA avg@8 | LoRA avg@8 | NoRA pass@8 | LoRA pass@8 |\n|---:|---:|---:|---:|---:|\n"
    reference_evals = {step: (avg, pas) for step, avg, pas in evaluation(reference)}
    for step, avg, pas in evaluation(run):
        base_avg, base_pas = reference_evals[step]
        text += f"| {step} | {pct(avg)} | {pct(base_avg)} | {pct(pas)} | {pct(base_pas)} |\n"
    text += "\n### Interpretation and next hypothesis\n\nThe early acceleration is a promising result even though the final evaluation does not beat the reference. Step-zero scores differ despite B=0 and unchanged initial logits; these are stochastic evaluation draws, not different starting weights. NoRA's positive signal is faster improvement in both training correctness and early held-out evaluation.\n\n"
    text += "The later plateau is an observation, not evidence that NoRA is its cause. A fixed rank-one update is one possible constraint; data difficulty, truncation and optimization dynamics are alternative explanations. This run does not identify the cause.\n\n"
    text += "**Next hypothesis: periodic merge/reset.** Merge the learned BA update into the backbone, then initialize a fresh adapter with NoRA-init and B=0. This preserves the policy at the reset boundary in exact arithmetic while allowing the accumulated update across cycles to exceed rank one. Test whether the early learning speed returns after reset and whether later reward and AIME25 accuracy improve. Optimizer-state reset and the merge interval must be explicit experimental settings. Existing exploratory merge runs also changed the optimizer and algorithm, so they do not validate this hypothesis. No new merge experiment has been launched.\n"
    return text


def diagnosis_plot():
    path = BOOK / "data/truncation-analysis.json"
    if not path.exists():
        return
    evidence = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for run, color in zip(evidence, COLORS):
        rows = run["evaluations"]
        steps = [r["step"] for r in rows]
        values = [
            [r["truncated"] / r["responses"] for r in rows],
            [r["avg_tokens"] for r in rows],
            [
                (r["correct"] - r["truncated_correct"]) / max(r["responses"] - r["truncated"], 1)
                for r in rows
            ],
        ]
        for ax, y in zip(axes, values):
            ax.plot(steps, y, color=color, label=run["label"], marker="o", lw=2)
    for ax, title in zip(
        axes,
        [
            "AIME25 truncation rate",
            "Mean evaluation response tokens",
            "Correctness among completed responses",
        ],
    ):
        ax.set(title=title, xlabel="Training step")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("After step 60 · more unfinished answers", fontsize=16, fontweight="bold")
    save(fig, "truncation-diagnosis")


def main():
    for d in ["experiments", "figures"]:
        (BOOK / d).mkdir(exist_ok=True)
    diagnosis_plot()
    registry = json.loads((BOOK / "data/registry.json").read_text())
    runs = [json.loads((BOOK / "data" / f"{r['id']}.json").read_text()) for r in registry]
    for run in runs:
        if run["metrics"]:
            run_plot(run)
        experiment_page(run)
    comparison(
        runs,
        {"primary", "reference"},
        "comparison-primary",
        "On-policy learning · Base model references",
    )
    comparison(runs, {"historical"}, "comparison-history", "Historical methods · settings differ")
    comparison(
        runs,
        {"confounded"},
        "comparison-confounded",
        "Earlier trials · optimizer and loss confounds",
    )
    nora_runs = [
        run
        for run in runs
        if run["config"].get("trainer.policy.model.lora.init_method") == "nora_init"
        or run["run_id"] == "qwen3-4b-base-grpo-lora-r1-blog-20260923-01"
    ]
    comparison(
        nora_runs, {"ablation", "primary"}, "comparison-nora", "NoRA-init · faster early learning"
    )
    text = """---
title: UNORL research book
---

# UNORL

**Resource-efficient, on-policy reinforcement learning.** This book connects each experiment to its configuration, measured learning curves, and evaluation results. It builds from small versioned data snapshots; weights and generated solution text stay outside the repository.

The present evidence supports learning with rank-1 LoRA and single-rollout REINFORCE using AdamW. At step 100, batch-normalized REINFORCE reaches AIME25 avg@8 **17.9%** and pass@8 **30.0%**, versus vanilla's **14.6% / 26.7%** and rank-1 GRPO's **20.0% / 43.3%**. These are single-run observations on 30 questions, not a reliable ranking of small differences. Earlier SGD trials are explicitly separated because they do not isolate algorithm choice.

## Main comparison

GRPO uses 32 prompts × 8 responses; REINFORCE uses 256 prompts × 1 response. Both generate 256 responses per update. The full-parameter reference uses different optimizer settings; all records expose their settings below.

"""
    text += figure(
        "figures/comparison-primary.svg",
        "Google Material palette. Evaluation points are unsmoothed; training curves use a trailing 10-update mean with raw values faintly shown.",
    )
    text += "\n## NoRA-init: faster early learning\n\nThe completed [NoRA-init trial](experiments/qwen3-4b-base-grpo-lora-r1-nora-init-20260928-01.md) reaches **33.8%** mean training correctness in steps 21–40, versus **23.7%** with standard rank-1 LoRA. Step-20 AIME25 avg@8 is **9.2% versus 4.2%**. Later training correctness plateaus near 38–39%; final avg@8 / pass@8 is **17.9% / 36.7%**, versus **20.0% / 43.3%**. Initialization and alpha differ together. The early acceleration motivates testing merge/reset; the cause of the plateau and benefit of merging remain unproven.\n\n"
    text += figure("figures/comparison-nora.svg", "NoRA-init and standard rank-1 LoRA GRPO.")
    text += "\n## All retained experiment results\n\nFinal columns use the last recorded AIME25 evaluation, whose step is shown separately from the last training step. A dash means missing evidence, not zero accuracy. Smoke tests are excluded.\n\n"
    csvrows = []
    for group, title in GROUPS.items():
        text += f"### {title}\n\n| Experiment | Last train step | Last eval step | Avg@8 | Pass@8 | Optimizer |\n|---|---:|---:|---:|---:|---|\n"
        for r in runs:
            if r["group"] != group:
                continue
            ev = evaluation(r)
            step, avg, pas = ev[-1] if ev else (None, None, None)
            last = r["metrics"][-1]["step"] if r["metrics"] else None
            text += f"| [{r['title']}](experiments/{r['run_id']}.md) | {last if last is not None else '—'} | {step if step is not None else '—'} | {pct(avg)} | {pct(pas)} | {r['optimizer']} |\n"
            csvrows.append(
                dict(
                    run_id=r["run_id"],
                    group=group,
                    last_train_step=last,
                    last_eval_step=step,
                    avg_at_8=avg,
                    pass_at_8=pas,
                )
            )
        text += "\n"
    text += figure(
        "figures/comparison-history.svg",
        "Earlier research used different objectives, model variants, and response budgets. These curves provide context, not a controlled ranking.",
    )
    text += figure(
        "figures/comparison-confounded.svg",
        "Earlier single-rollout trials used stateless SGD and sometimes a different loss reduction. Their failures do not establish that REINFORCE fails with AdamW.",
    )
    text += "\n## Post-step-60 diagnosis\n\nThe [truncation analysis](notes/batchnorm-after60.md) examines the loss of question coverage and specifies a controlled follow-up trial.\n\n## Research directions and next questions\n\nThe [2025–2026 LoRA training investigation](notes/lora-training-2025-2026.md) compares LoRA-FA, LoFT, recent optimizer-state research, and merge/reset designs. It separates published evidence from proposed UNORL experiments.\n\n1. Final-layer LoRA: the last-half trial completed; measure actual activation/peak memory savings and investigate fewer layers.\n2. [NoRA initialization](notes/nora.md): the trial completed with promising early acceleration. Next, test periodic merge/reset as a possible way to sustain learning; this is a hypothesis, not an executed follow-up.\n3. Test QLoRA separately; this direction remains untested.\n\nThe current scope is on-policy learning; small batches and single-rollout use remain central.\n\n[Measurement conventions](methods.md) · [Build and publish](publishing.md) · [Download comparison data](data/comparison.csv)\n"
    (BOOK / "index.md").write_text(text)
    with (BOOK / "data/comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(csvrows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(csvrows)
    toc = [{"file": "research/index.md"}, {"file": "research/methods.md"}]
    for group, title in GROUPS.items():
        toc.append(
            {
                "title": title,
                "children": [
                    {"file": f"research/experiments/{r['run_id']}.md", "title": r["title"]}
                    for r in runs
                    if r["group"] == group
                ],
            }
        )
    toc.extend(
        [
            {"file": "research/notes/nora.md"},
            {"file": "research/notes/lora-training-2025-2026.md"},
            {"file": "research/notes/batchnorm-after60.md"},
            {"file": "research/publishing.md"},
        ]
    )
    cfg = {
        "version": 1,
        "project": {
            "title": "UNORL",
            "license": "MIT",
            "description": "A research book on resource-efficient on-policy reinforcement learning.",
            "exclude": [
                "runs/**",
                "models/**",
                "data/**",
                "node_modules/**",
                ".venv-docs/**",
                ".deps/**",
                ".cache/**",
                ".tmp/**",
            ],
            "toc": toc,
        },
        "site": {"template": "book-theme", "title": "UNORL", "options": {"logo_text": "UNORL"}},
    }
    (ROOT / "myst.yml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"Built {len(runs)} experiment pages and comparison figures")


if __name__ == "__main__":
    main()
