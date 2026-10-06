"""Exercise state semantics with real PEFT factors/Adam and mocked DP collectives."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from unorl.guided_reset_worker import GuidedResetPolicyWorker
from unorl.low_resource import adapter_layers
from unorl.relora_worker import ReLoRAPolicyWorker


@pytest.mark.parametrize("branch", ["standard", "random", "guided"])
def test_boundary_preserves_function_scheduler_and_expected_adam_state(tmp_path, branch):
    torch.manual_seed(42)
    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                tie_word_embeddings=True,
            )
        ),
        LoraConfig(r=1, lora_alpha=32, target_modules="all-linear", lora_dropout=0),
    )
    worker = object.__new__(GuidedResetPolicyWorker)
    worker.model = SimpleNamespace(model=model)
    worker.cfg = SimpleNamespace(
        seed=42,
        boundary_probe_width=4,
        boundary_token_chunk=2,
        boundary_branch=branch,
        relora_probe_response_tokens=2,
        export_path=str(tmp_path / "exports"),
    )
    worker.optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1.5e-5,
        weight_decay=0,
    )
    worker.scheduler = torch.optim.lr_scheduler.LambdaLR(worker.optimizer, lambda _: 1)
    ids = torch.tensor([[1, 2, 3, 4]])
    model(ids).logits.square().mean().backward()
    worker.optimizer.step()
    worker.optimizer.zero_grad(set_to_none=True)
    for state in worker.optimizer.state.values():
        state["step"].fill_(40)
    worker.scheduler.last_epoch = 40
    worker._factor_history = {}
    data = {
        "sequences": ids,
        "attention_mask": torch.ones_like(ids),
        "response_mask": torch.ones(1, 2),
    }
    before = model(ids).logits.detach().clone()
    params = {name: p.detach().clone() for name, p in model.named_parameters()}
    moments = copy.deepcopy(worker.optimizer.state_dict())
    rng = torch.get_rng_state().clone()

    def native_calibration(self, data):
        self.model.model(data["sequences"]).logits.square().mean().backward()

    with (
        patch("unorl.guided_reset_worker.dist.get_rank", return_value=0),
        patch("unorl.guided_reset_worker.dist.get_world_size", return_value=1),
        patch("unorl.guided_reset_worker.dist.broadcast_object_list"),
        patch("unorl.guided_reset_worker.dist.all_reduce"),
        patch("unorl.relora_worker.dist.get_rank", return_value=0),
        patch("unorl.guided_reset_worker.torch.cuda.current_device", return_value=0),
        patch("unorl.guided_reset_worker.torch.cuda.get_rng_state", return_value=torch.zeros(1)),
        patch("unorl.guided_reset_worker.torch.cuda.set_rng_state"),
        patch.object(ReLoRAPolicyWorker, "forward_backward", new=native_calibration),
    ):
        worker._calibrate(data, (), {})

    torch.testing.assert_close(model(ids).logits, before, atol=2e-6, rtol=2e-5)
    assert worker.scheduler.last_epoch == 40
    assert worker.optimizer.param_groups[0]["lr"] == 1.5e-5
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in model.parameters())
    if branch == "standard":
        for name, p in model.named_parameters():
            assert torch.equal(p, params[name])
        restored = worker.optimizer.state_dict()
        assert restored["param_groups"] == moments["param_groups"]
        for index, state in moments["state"].items():
            for key, value in state.items():
                assert torch.equal(restored["state"][index][key], value)
    else:
        assert not worker.optimizer.state
        assert worker._pending_base_sync
        assert all(
            not layer.lora_B["default"].weight.count_nonzero() for _, layer in adapter_layers(model)
        )
        assert all(len(history) == 1 for history in worker._factor_history.values())
