"""Check response grouping, paired uncertainty and incomplete trial reporting."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "refresh_analysis", Path(__file__).resolve().parents[1] / "scripts/research/analyze_refresh.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def evaluation(counts, step=100):
    return {
        "step": step,
        "benchmark": "aime25",
        "questions": [
            {"prompt_sha256": str(i), "samples": 8, "correct": count}
            for i, count in enumerate(counts)
        ],
    }


def test_paired_metrics_keep_question_grouping_and_reject_incomplete_evaluation():
    left, right = evaluation([0, 4, 8]), evaluation([8, 4, 0])
    assert analysis.paired_questions(left, right, "avg@8").tolist() == [1, 0, -1]
    assert analysis.paired_questions(left, right, "pass@8").tolist() == [1, 0, -1]
    right["questions"].reverse()
    assert analysis.paired_questions(left, right, "avg@8").tolist() == [1, 0, -1]
    right["questions"][0]["samples"] = 7
    with pytest.raises(ValueError, match="eight"):
        analysis.paired_questions(left, right, "avg@8")
    with pytest.raises(ValueError, match="identical"):
        analysis.paired_questions(left, evaluation([8, 4]), "avg@8")


def test_bootstrap_degenerate_and_mixed_differences():
    result = analysis.bootstrap_difference([0.125] * 30)
    assert result["candidate_minus_control"] == 0.125
    assert result["question_bootstrap_95_interval"] == [0.125, 0.125]
    mixed = analysis.bootstrap_difference([-1, 0, 1] * 10)
    assert mixed["candidate_minus_control"] == 0
    assert (
        mixed["question_bootstrap_95_interval"][0] < 0 < mixed["question_bootstrap_95_interval"][1]
    )
    assert mixed == analysis.bootstrap_difference([-1, 0, 1] * 10)


def test_partial_windows_and_initial_adjustment_do_not_fabricate_completion():
    control = {
        "config": {"trainer.policy.optimizer_config.lr": 1.5e-5},
        "metrics": [{"step": 101, "metrics": {"reward/mean_positive_reward": 0.4}}],
        "evaluation_question_scores": [evaluation([0, 4, 8]), evaluation([0, 4, 8], 120)],
    }
    pending = analysis.analyze(control, None)
    assert pending["status"] == "candidate not started"
    assert pending["evaluations"] == {}
    candidate = {
        "config": dict(control["config"]),
        "metrics": [{"step": 101, "metrics": {"reward/mean_positive_reward": 0.5}}],
        "evaluation_question_scores": [evaluation([1, 5, 8]), evaluation([2, 6, 8], 120)],
    }
    report = analysis.analyze(control, candidate)
    assert report["windows"]["101-120"]["paired_updates"] == 1
    assert report["windows"]["101-120"]["complete"] is False
    gain = report["evaluations"]["120"]["avg@8"]["difference_in_improvement_from_step100"]
    assert gain["candidate_minus_control"] == pytest.approx(1 / 12)
    candidate["config"]["trainer.policy.optimizer_config.lr"] = 1e-3
    with pytest.raises(ValueError, match="recipe differs"):
        analysis.analyze(control, candidate)


def test_base_comparison_exposes_recipe_differences_and_incomplete_final_window():
    from scripts.research.analyze_refresh_base import analyze

    control = {
        "run_id": "historical",
        "config": {"trainer.ckpt_interval": 100},
        "metrics": [{"step": 91, "metrics": {"reward/mean_positive_reward": 0.4}}],
        "evaluation_question_scores": [evaluation([0, 4, 8], 0), evaluation([1, 5, 8], 80)],
    }
    candidate = {
        "run_id": "refresh",
        "config": {"trainer.ckpt_interval": 20},
        "metrics": [{"step": 91, "metrics": {"reward/mean_positive_reward": 0.5}}],
        "evaluation_question_scores": [evaluation([1, 5, 8], 0), evaluation([2, 6, 8], 80)],
    }
    report = analyze(control, candidate)
    window = report["windows"]["91-100"]
    assert window["updates"] == 1
    assert window["complete"] is False
    assert window["metrics"]["reward/mean_positive_reward"]["candidate"] == 0.5
    assert report["config_differences"]["trainer.ckpt_interval"] == {
        "control": 100,
        "candidate": 20,
    }
    assert "100" not in report["evaluations"]
    gain = report["evaluations"]["80"]["avg@8"]["difference_in_improvement_from_step0"]
    assert gain["candidate_minus_control"] == 0
