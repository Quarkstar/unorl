"""Check the controlled GRPO adapter placement against the full-layer recipe."""

import json
import re
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from unorl.nora import normalize_lora_initialization

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "baseline_name,trial_name",
    [
        ("qwen3-4b-base-grpo-lora-r1-blog", "qwen3-4b-base-grpo-lora-r1-last18"),
        ("qwen3-4b-base-grpo-lora-r1-nora-init", "qwen3-4b-base-grpo-lora-r1-nora-init-last18"),
    ],
)
def test_last_half_lora_targets_only_layers_18_to_35(baseline_name, trial_name):
    baseline = json.loads((ROOT / "configs" / f"{baseline_name}.json").read_text())
    trial = json.loads((ROOT / "configs" / f"{trial_name}.json").read_text())
    exclude = trial.pop("trainer.policy.model.lora.exclude_modules")
    assert trial == baseline
    assert trial["trainer.policy.model.lora.target_modules"] == "all-linear"

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=36,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
    )
    model = get_peft_model(
        Qwen3ForCausalLM(config),
        LoraConfig(
            r=1,
            lora_alpha=trial["trainer.policy.model.lora.alpha"],
            target_modules="all-linear",
            exclude_modules=exclude,
        ),
    )
    if trial.get("trainer.policy.model.lora.init_method") == "nora_init":
        audit = normalize_lora_initialization(model)
        assert audit["modules"] == 18 * 7
        for name, parameter in model.named_parameters():
            if "lora_A" in name:
                assert ((parameter == 1) | (parameter == -1)).all()
            if "lora_B" in name:
                assert torch.count_nonzero(parameter) == 0
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    layer_counts = {layer: 0 for layer in range(36)}
    for name in trainable:
        layer = re.search(r"\.layers\.(\d+)\.", name)
        assert layer is not None, name
        layer_counts[int(layer.group(1))] += 1

    assert all(layer_counts[layer] == 0 for layer in range(18))
    assert all(layer_counts[layer] == 14 for layer in range(18, 36))
