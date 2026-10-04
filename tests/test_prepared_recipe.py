"""Protect matched rollout/training budgets against unintended recipe changes."""

import json
from pathlib import Path


def test_prepared_recipe_matches_historical_standard_except_method_and_checkpoints():
    root = Path(__file__).resolve().parents[1]
    standard = json.loads((root / "configs/qwen3-4b-base-grpo-lora-r1-blog.json").read_text())
    prepared = json.loads((root / "configs/qwen3-4b-base-grpo-prepared-r1.json").read_text())
    shared = {
        key: value
        for key, value in prepared.items()
        if not key.startswith(("trainer.prepared_", "trainer.relora_"))
    }
    shared["trainer.ckpt_interval"] = standard["trainer.ckpt_interval"]
    assert shared == standard
    assert prepared["trainer.max_training_steps"] == 100
    assert prepared["trainer.placement.policy_num_gpus_per_node"] == 8
    assert prepared["trainer.prepared_window_updates"] == 20
    assert prepared["trainer.relora_first_merge_step"] == 40
    assert prepared["trainer.relora_merge_interval"] == 40
    assert prepared["trainer.relora_restart_warmup_updates"] == 0
