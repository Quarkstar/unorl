"""CPU-only integrity checks for publication data and the renamed launch interface."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "research/data/registry.json").read_text())


@pytest.mark.parametrize("record", REGISTRY, ids=lambda r: r["id"])
def test_published_evaluations_agree_with_response_counts(record):
    data = json.loads((ROOT / "research/data" / f"{record['id']}.json").read_text())
    by_step = {row["step"]: row["metrics"] for row in data["metrics"]}
    assert len(by_step) == len(data["metrics"])
    assert "smoke" not in data["run_id"]
    assert (ROOT / "research/experiments" / f"{record['id']}.md").exists()
    for count in data["evaluation_counts"]:
        m = by_step[count["step"]]
        prefix = f"eval/{count['benchmark']}/"
        assert count["correct_responses"] / count["responses"] == pytest.approx(
            m[prefix + "mean_positive_reward"]
        )
        if prefix + "pass_at_8" in m:
            assert count["solved_questions"] / count["questions"] == pytest.approx(
                m[prefix + "pass_at_8"]
            )


def test_dry_run_is_portable_and_uses_unorl():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train.py"), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    resolved = json.loads(result.stdout)
    assert resolved["command"][2] == "unorl.reinforce_adamw_train"
    assert resolved["config"]["environment.env_class"] == "unorl_math"
    assert resolved["config"]["trainer.project_name"] == "unorl"
    assert resolved["config"]["trainer.policy.model.path"] == str(ROOT / "models/Qwen3-4B-Base")


def test_baseline_prompt_is_unchanged():
    spec = importlib.util.spec_from_file_location("prompts", ROOT / "unorl/prompts.py")
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)

    class Tokenizer:
        chat_template = "ORIGINAL_TEMPLATE"

    original = Tokenizer()
    wrapped = prompts.system_tokenizer(original)
    expected = (
        '{% set messages = [{"role": "system", "content": '
        + json.dumps(r"Please reason step by step. End with Answer: \boxed{your final answer}.")
        + "}] + messages %}ORIGINAL_TEMPLATE"
    )
    assert wrapped.chat_template == expected
    assert original.chat_template == "ORIGINAL_TEMPLATE"


def test_relora_cleanup_retains_evaluations_and_diagnostics(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "relora_comparison", ROOT / "scripts/run_relora_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deleted = [
        "checkpoints/model.pt",
        "exports/global_step_100/policy/model.safetensors",
        "exports/global_step_100/critic/model.safetensors",
        "relora-rank-factors.pt",
    ]
    retained = [
        "exports/aime25/dumped_evals/global_step_100_evals/aime25.jsonl",
        "exports/aime25/dumped_evals/global_step_100_evals/aggregated_results.jsonl",
        "metrics.jsonl",
        "rank-diagnostics.jsonl",
        "exports/global_step_notes/report.txt",
    ]
    for name in deleted + retained:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("retained-or-deleted")
    removed = module.cleanup_artifacts(tmp_path)
    assert len(removed) == 4
    assert all(not (tmp_path / name).exists() for name in deleted)
    assert all((tmp_path / name).read_text() == "retained-or-deleted" for name in retained)
    assert module.cleanup_artifacts(tmp_path) == []
