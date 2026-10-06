import copy
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from unorl.update_cap_worker import UpdateCapPolicyWorker


def test_cap_keeps_native_adam_state_and_scheduler_while_limiting_applied_update(tmp_path):
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
            )
        ),
        LoraConfig(r=1, lora_alpha=32, target_modules="all-linear"),
    )
    worker = object.__new__(UpdateCapPolicyWorker)
    worker.model = SimpleNamespace(model=model)
    worker.cfg = SimpleNamespace(export_path=str(tmp_path / "exports"), max_training_steps=100)
    worker.optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    worker.scheduler = torch.optim.lr_scheduler.LambdaLR(worker.optimizer, lambda _: 1)
    worker._previous_delta = None
    worker._before_step_factors = worker._capture_factors()
    model(torch.tensor([[1, 2, 3, 4]])).logits.square().mean().backward()
    worker.optimizer.step()
    worker.scheduler.step()
    native = copy.deepcopy(worker.optimizer.state_dict())
    worker._update_budget = 0.0001
    with (
        patch("unorl.update_cap_worker.dist.get_rank", return_value=0),
        patch("unorl.relora_refresh_worker.dist.get_rank", return_value=0),
        patch("unorl.update_cap_worker.dist.broadcast"),
        patch("unorl.update_cap_worker.torch.cuda.current_device", return_value="cpu"),
    ):
        metrics = worker._measure_delta()
    assert 0 < metrics["update_cap/step_multiplier"] < 1
    assert metrics["updates/effective_delta_l2"] <= 0.0001 * (1 + 1e-5) + 1e-8
    assert worker.scheduler.last_epoch == 1
    assert worker.optimizer.param_groups[0]["lr"] == 0.01
    actual = worker.optimizer.state_dict()
    assert actual["param_groups"] == native["param_groups"]
    for index, state in native["state"].items():
        for key, value in state.items():
            assert torch.equal(actual["state"][index][key], value)
