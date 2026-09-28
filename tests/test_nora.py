"""Check NoRA's geometry, initial policy, training and native export contract."""

import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from unorl.nora import normalize_lora_initialization


def tiny_model(rank):
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
    )
    return get_peft_model(
        Qwen3ForCausalLM(config),
        LoraConfig(r=rank, lora_alpha=rank, target_modules="all-linear"),
    )


@pytest.mark.parametrize("rank", [1, 4])
def test_normalized_columns_preserve_policy_and_learn(rank, tmp_path):
    model = tiny_model(rank).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    before = model(tokens).logits.detach()
    audit = normalize_lora_initialization(model)
    assert audit["modules"] == 14
    torch.testing.assert_close(model(tokens).logits, before, rtol=0, atol=0)
    for name, value in model.named_parameters():
        if "lora_A" in name:
            torch.testing.assert_close(value.norm(dim=0), torch.ones(value.shape[1]))
            if rank == 1:
                assert ((value == 1) | (value == -1)).all()
    assert 0.35 < audit["positive_A_entries"] / audit["materialized_A_entries"] < 0.65

    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad], lr=1e-3
    )
    for _ in range(2):
        optimizer.zero_grad()
        model(tokens, labels=tokens).loss.backward()
        optimizer.step()
    assert any(
        value.grad.abs().sum() > 0 for name, value in model.named_parameters() if "lora_A" in name
    )
    trained = model(tokens).logits.detach()
    assert not torch.equal(trained, before)
    model.save_pretrained(tmp_path)
    model.load_adapter(tmp_path, adapter_name="reload")
    model.set_adapter("reload")
    torch.testing.assert_close(model(tokens).logits, trained)
    merged = model.merge_and_unload()
    torch.testing.assert_close(merged(tokens).logits, trained, rtol=1e-4, atol=1e-6)


def test_reject_normalization_of_trained_adapter():
    model = tiny_model(1)
    with torch.no_grad():
        next(value for name, value in model.named_parameters() if "lora_B" in name).fill_(1)
    with pytest.raises(ValueError, match="zero B"):
        normalize_lora_initialization(model)


def test_nora_profile_changes_only_initialization_and_scaling():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "configs/qwen3-4b-base-grpo-lora-r1-blog.json").read_text())
    trial = json.loads((root / "configs/qwen3-4b-base-grpo-lora-r1-nora-init.json").read_text())
    assert trial.pop("trainer.policy.model.lora.init_method") == "nora_init"
    assert trial["trainer.policy.model.lora.alpha"] == 1
    trial["trainer.policy.model.lora.alpha"] = 32
    assert trial == baseline


def test_worker_wrapper_initializes_before_native_sharding(tmp_path):
    from unorl.nora_worker import NoRAInitModelWrapper

    model = tiny_model(1).merge_and_unload()
    model.save_pretrained(tmp_path)
    wrapped = NoRAInitModelWrapper(
        str(tmp_path),
        use_flash_attention_2=False,
        bf16=False,
        lora_rank=1,
        lora_alpha=1,
        lora_init_method="nora_init",
        target_modules="all-linear",
        remove_microbatch_padding=False,
        use_torch_compile=False,
    )
    for name, value in wrapped.model.named_parameters():
        if "lora_A" in name:
            assert ((value == 1) | (value == -1)).all()
        if "lora_B" in name:
            assert torch.count_nonzero(value) == 0
    assert wrapped.model.peft_config["default"].lora_alpha == 1
    assert wrapped.model.peft_config["default"].init_lora_weights is True
