"""Compare matched ReLoRA boundaries using portable published metric snapshots."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "research"
IDS = {
    "Standard LoRA": "qwen3-4b-base-grpo-lora-r1-blog-20260923-01",
    "Merge + restart ramp": "qwen3-4b-base-grpo-relora-r1-warmup5-20261002-01",
    "Merge + constant LR": "qwen3-4b-base-grpo-relora-r1-warmup0-20261002-01",
}
COLORS = ["#2196F3", "#FF9800", "#009688"]
WINDOWS = [(1, 20), (21, 40), (41, 45), (46, 60), (41, 60), (61, 80), (81, 90), (91, 100)]
KEYS = [
    "reward/mean_positive_reward",
    "policy/policy_entropy",
    "generate/avg_num_tokens",
    "policy/grad_norm",
    "policy/rollout_train_logprobs_abs_diff_mean",
]


def main():
    data = {
        name: json.loads((ROOT / "data" / f"{rid}.json").read_text()) for name, rid in IDS.items()
    }
    rows = {name: {r["step"]: r["metrics"] for r in run["metrics"]} for name, run in data.items()}
    reference = json.loads((ROOT / "data/standard-lora-final-update.json").read_text())
    summary = {
        "windows": {},
        "update_alignment": {},
        "final_delta_l2": {"Standard LoRA": reference["total_adapter_delta_l2"]},
    }
    for name, metrics in rows.items():
        summary["windows"][name] = {
            f"{lo}-{hi}": {
                key: float(np.mean([metrics[s][key] for s in range(lo, hi + 1)])) for key in KEYS
            }
            for lo, hi in WINDOWS
        }
        if name != "Standard LoRA":
            run = data[name]
            final = next(r for r in run["rank_diagnostics"] if r["step"] == 100)
            summary["final_delta_l2"][name] = float(
                np.sqrt(sum(layer["update_l2"] ** 2 for layer in final["layers"].values()))
            )
            audits = run["merge_audits"]
            ranks = run["rank_diagnostics"]
            first = next(r for r in ranks if r["step"] == 40)
            second = next(r for r in ranks if r["step"] == 80)
            norm1, norm2 = [a["relora/delta_l2"] for a in audits]
            energy1 = sum(layer["update_l2"] ** 2 for layer in first["layers"].values())
            energy2 = sum(layer["update_l2"] ** 2 for layer in second["layers"].values())
            assert np.isclose(energy1, norm1**2, rtol=1e-5)
            dot = (energy2 - norm1**2 - norm2**2) / 2
            summary["update_alignment"][name] = {
                "cycle1_l2": norm1,
                "cycle2_l2": norm2,
                "inferred_global_frobenius_cosine_cycles1_and2": dot / (norm1 * norm2),
                "method": "Polarization identity from accumulated spectra and per-cycle delta norms; approximate due to float32 diagnostics",
            }
    (ROOT / "data/relora-boundary-analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "svg.hashsalt": "unorl-relora-boundaries",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    specs = [
        (KEYS[0], "Training correctness (%)", 100),
        (None, "Correctness gap versus standard (percentage points)", 100),
        (KEYS[1], "Policy entropy", 1),
        (KEYS[2], "Mean response tokens", 1),
        (KEYS[3], "Gradient norm (before clipping)", 1),
        ("eval/aime25/avg@8", "AIME25 avg@8 (%)", 100),
    ]
    steps = np.arange(1, 101)
    centers = np.arange(3, 101, 5)
    for ax, (key, title, scale) in zip(axes.flat, specs):
        for color, (name, metrics) in zip(COLORS, rows.items()):
            if key == "eval/aime25/avg@8":
                available = sorted(s for s, m in metrics.items() if key in m)
                ax.plot(
                    available,
                    [metrics[s][key] * scale for s in available],
                    marker="o",
                    color=color,
                    label=name,
                )
            else:
                if key is None:
                    values = np.array(
                        [
                            (metrics[s][KEYS[0]] - rows["Standard LoRA"][s][KEYS[0]]) * scale
                            for s in steps
                        ]
                    )
                else:
                    values = np.array([metrics[s][key] * scale for s in steps])
                ax.plot(steps, values, color=color, alpha=0.12, linewidth=0.7)
                ax.plot(
                    centers,
                    values.reshape(20, 5).mean(axis=1),
                    color=color,
                    label=name,
                    linewidth=2,
                )
        for boundary in [40.5, 80.5]:
            ax.axvline(boundary, color="#607D8B", linestyle="--", alpha=0.7)
        ax.set(title=title, xlabel="Training step", xlim=(0, 101))
        if key is None:
            ax.axhline(0, color="#607D8B", linewidth=0.8)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "ReLoRA: divergence follows the first reset; later reward largely catches up\nBold lines: disjoint five-step means. Dashed lines: merges after updates 40 and 80.",
        fontsize=12,
    )
    for suffix in ("svg", "png"):
        fig.savefig(ROOT / "figures" / f"relora-boundary-analysis.{suffix}", dpi=180)
    plt.close(fig)
    print("Saved ReLoRA boundary analysis, numeric windows and comparison plots")


if __name__ == "__main__":
    main()
