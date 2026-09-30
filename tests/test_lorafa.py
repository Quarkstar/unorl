"""Check LoRA-FA geometry, frozen weights, activation storage and export."""

import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from unorl.lorafa import (
    correct_gradients,
    correction_factors,
    freeze_lora_a,
    rank_one_correction,
)


def tiny_model(rank=1):
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
                tie_word_embeddings=True,
            )
        ),
        LoraConfig(r=rank, lora_alpha=32, target_modules="all-linear"),
    )


def test_correction_matches_projected_full_weight_gradient():
    torch.manual_seed(5)
    a = torch.randn(1, 16)
    g = torch.randn(8, 16)
    scale = 32.0
    grad_b = scale * g @ a.T
    corrected = grad_b * rank_one_correction(a, scale, regularization=0)
    weight_grad = scale * corrected @ a
    projector = a.T @ torch.linalg.pinv(a @ a.T) @ a
    torch.testing.assert_close(weight_grad, g @ projector)
    # Rescaling A and inversely rescaling B must not change this projection.
    other_a = 7 * a
    other_grad = scale * g @ other_a.T
    other_weight_grad = (
        scale * other_grad * rank_one_correction(other_a, scale, regularization=0) @ other_a
    )
    torch.testing.assert_close(other_weight_grad, weight_grad)


def test_only_b_learns_and_native_adapter_round_trip(tmp_path):
    model = tiny_model().eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    initial_logits = model(tokens).logits.detach()
    assert freeze_lora_a(model) == 14
    torch.testing.assert_close(model(tokens).logits, initial_logits, rtol=0, atol=0)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    factors = correction_factors(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    for _ in range(3):
        model(tokens, labels=tokens).loss.backward()
        assert correct_gradients(model, factors) == 14
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
            assert parameter.grad is None
            assert parameter not in optimizer.state
    assert len(optimizer.state) == 14
    trained = model(tokens).logits.detach()
    assert not torch.equal(initial_logits, trained)
    model.save_pretrained(tmp_path)
    from peft import PeftModel

    reloaded = tiny_model().merge_and_unload()
    reloaded = PeftModel.from_pretrained(reloaded, tmp_path).eval()
    torch.testing.assert_close(reloaded(tokens).logits, trained)
    torch.testing.assert_close(
        model.merge_and_unload()(tokens).logits, trained, rtol=1e-4, atol=1e-6
    )


def test_freezing_a_removes_wide_adapter_input_storage():
    model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(16, 8, bias=False)),
        LoraConfig(r=1, lora_alpha=32, target_modules=["0"]),
    )
    x = torch.randn(2, 4, 16, requires_grad=True)

    def saved_inputs():
        saved = []

        def pack(tensor):
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            model(x).sum().backward()
        return sum(t.data_ptr() == x.data_ptr() for t in saved)

    assert saved_inputs() > 0
    model.zero_grad()
    x.grad = None
    freeze_lora_a(model)
    assert saved_inputs() == 0
    # Earlier adapters still need input gradients; A freezing must not detach x.
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_profile_matches_reference_except_method_selector():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "configs/qwen3-4b-base-grpo-lora-r1-blog.json").read_text())
    trial = json.loads((root / "configs/qwen3-4b-base-grpo-lorafa-r1.json").read_text())
    assert trial.pop("trainer.policy.model.lora.init_method") == "lorafa"
    assert trial == baseline


def test_rejects_unsupported_rank_and_invalid_a():
    with pytest.raises(ValueError, match="rank one"):
        freeze_lora_a(tiny_model(rank=2))
    with pytest.raises(ValueError, match="nonzero A"):
        rank_one_correction(torch.zeros(1, 16), 32)


def test_worker_wrapper_freezes_before_fsdp(tmp_path):
    from unorl.lorafa_worker import LoRAFAWrapper

    tiny_model().merge_and_unload().save_pretrained(tmp_path)
    wrapper = LoRAFAWrapper(
        str(tmp_path),
        use_flash_attention_2=False,
        bf16=False,
        lora_rank=1,
        lora_alpha=32,
        lora_init_method="lorafa",
        target_modules="all-linear",
        remove_microbatch_padding=False,
        use_torch_compile=False,
    )
    for name, parameter in wrapper.model.named_parameters():
        assert parameter.requires_grad == (".lora_B." in name)
    assert wrapper.model.peft_config["default"].init_lora_weights is True
