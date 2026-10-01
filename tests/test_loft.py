"""LoFT-simple alternation, moving-coordinate history and checkpoint semantics."""

import copy
import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from unorl.loft import LoFTSimpleAdamW


def tiny_model():
    torch.manual_seed(42)
    return get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                tie_word_embeddings=False,
            )
        ),
        LoraConfig(r=1, lora_alpha=1, target_modules="all-linear"),
    )


def test_alternation_with_both_moments_updated_and_frozen_backbone():
    model = tiny_model()
    optimizer = LoFTSimpleAdamW(model, lr=0.001)
    tokens = torch.tensor([[1, 2, 3, 4]])
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    for step in range(1, 5):
        previous = {n: p.detach().clone() for n, p in model.named_parameters()}
        model(tokens, labels=tokens).loss.backward()
        assert optimizer.calibrated_grad_norm() > 0
        optimizer.step()
        optimizer.zero_grad()
        active = "lora_B" if step % 2 else "lora_A"
        changed = []
        for n, p in model.named_parameters():
            if active not in n:
                torch.testing.assert_close(p, previous[n], rtol=0, atol=0)
            else:
                changed.append(not torch.equal(p, previous[n]))
            if not p.requires_grad:
                torch.testing.assert_close(p, initial[n], rtol=0, atol=0)
                assert p not in optimizer.state
        assert any(changed)
        assert all(s["step"] == step for s in optimizer.state.values())
        assert len(optimizer.state) == 28
        assert all("previous_other" in s for s in optimizer.state.values())


def test_checkpoint_reproduces_next_update_including_alternation_and_previous_factors():
    model = tiny_model()
    optimizer = LoFTSimpleAdamW(model, lr=0.001)
    tokens = torch.tensor([[1, 2, 3, 4]])
    model(tokens, labels=tokens).loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    clone = tiny_model()
    clone.load_state_dict(model.state_dict())
    restored = LoFTSimpleAdamW(clone, lr=0.001)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert optimizer.update_A == restored.update_A is True
    for current, opt in [(model, optimizer), (clone, restored)]:
        current(tokens, labels=tokens).loss.backward()
        opt.step()
        opt.zero_grad()
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, dict(clone.named_parameters())[name], rtol=0, atol=0)


def test_clipping_norm_uses_only_active_calibrated_gradients():
    model = tiny_model()
    opt = LoFTSimpleAdamW(model, lr=0.001)
    total = 0
    for p, (other, is_a) in opt.pairs.items():
        p.grad = torch.ones_like(p)
        if not is_a:
            inverse = torch.linalg.inv(other @ other.T + 1e-8 * torch.eye(1))
            total += (p.grad @ inverse).square().sum().item()
    assert opt.calibrated_grad_norm() == pytest.approx(total**0.5, rel=1e-6)


def test_loft_profile_and_wrapper_preserve_reference_initial_policy(tmp_path):
    from unorl.loft_worker import LoFTWrapper

    root = Path(__file__).resolve().parents[1]
    ref = json.loads((root / "configs/qwen3-4b-base-grpo-lora-r1-blog.json").read_text())
    trial = json.loads((root / "configs/qwen3-4b-base-grpo-loft-simple-r1.json").read_text())
    assert trial.pop("trainer.policy.model.lora.init_method") == "loft_simple"
    trial["trainer.policy.model.lora.alpha"] = 32
    assert trial == ref
    base = tiny_model().merge_and_unload().eval()
    base.save_pretrained(tmp_path)
    wrapped = LoFTWrapper(
        str(tmp_path),
        use_flash_attention_2=False,
        bf16=False,
        lora_rank=1,
        lora_alpha=1,
        lora_init_method="loft_simple",
        target_modules="all-linear",
        remove_microbatch_padding=False,
        use_torch_compile=False,
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(wrapped.model(tokens).logits, base(tokens).logits, rtol=0, atol=0)
    assert len(LoFTSimpleAdamW(wrapped.model, lr=0.001).pairs) == 28
