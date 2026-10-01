"""ReLoRA restart boundaries, same-coordinate merge control and rank diagnostics."""

import copy
import json
from pathlib import Path

import pytest
import torch
from test_loft import tiny_model

from unorl.low_resource import adapter_layers, copy_parameter, merge_and_reset
from unorl.relora import (
    accumulated_spectrum,
    distribution_shift,
    response_log_distribution,
    restart_multiplier,
    trajectory_prefix,
)
from unorl.relora_config import ReLoRATrainConfig, validate_relora


def test_restart_lr_update_order_and_checkpoint_resume():
    p = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([p], lr=0.001)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: restart_multiplier(k, 40, 5))
    applied = []
    for _ in range(44):
        applied.append(opt.param_groups[0]["lr"])
        p.grad = torch.ones_like(p)
        opt.step()
        sched.step()
        opt.zero_grad()
    assert applied[:40] == [0.001] * 40
    assert applied[40:44] == [0, 0.00025, 0.0005, 0.00075]
    assert sched.get_last_lr() == [0.001]
    new = torch.nn.Parameter(p.detach().clone())
    restored = torch.optim.AdamW([new], lr=0.001)
    replay = torch.optim.lr_scheduler.LambdaLR(restored, lambda k: restart_multiplier(k, 40, 5))
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    replay.load_state_dict(sched.state_dict())
    assert replay.last_epoch == 44
    for step in range(44, 85):
        assert replay.get_last_lr() == sched.get_last_lr()
        for parameter, optimizer, scheduler in [(p, opt, sched), (new, restored, replay)]:
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        torch.testing.assert_close(new, p, rtol=0, atol=0)
    assert restart_multiplier(80, 40, 0) == 1
    assert restart_multiplier(80, 40, 5) == 0
    for warmup in [-1, 1, 40]:
        with pytest.raises(ValueError):
            restart_multiplier(0, 40, warmup)


def test_fixed_a_merge_with_preserved_adam_reproduces_next_weight_update():
    model = tiny_model().eval()
    for _, layer in adapter_layers(model):
        layer.lora_A["default"].weight.requires_grad_(False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    for _ in range(3):
        model(tokens, labels=tokens).loss.backward()
        opt.step()
        opt.zero_grad()
    merged = copy.deepcopy(model)
    replay = torch.optim.AdamW(
        [p for p in merged.parameters() if p.requires_grad], lr=0.001, weight_decay=0
    )
    replay.load_state_dict(copy.deepcopy(opt.state_dict()))
    factors = {
        n: layer.lora_A["default"].weight.detach().clone() for n, layer in adapter_layers(merged)
    }
    merge_and_reset(merged, seed=200)
    for n, layer in adapter_layers(merged):
        copy_parameter(layer.lora_A["default"].weight, factors[n])
    torch.testing.assert_close(merged(tokens).logits, model(tokens).logits, rtol=1e-4, atol=1e-6)
    for current, optimizer in [(model, opt), (merged, replay)]:
        current(tokens, labels=tokens).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.testing.assert_close(merged(tokens).logits, model(tokens).logits, rtol=1e-4, atol=1e-6)
    assert all(s["step"] == 4 for s in replay.state.values())


def test_spectrum_detects_useful_rank_and_output_direction_collapse():
    a1, a2 = torch.tensor([[1.0, 0, 0]]), torch.tensor([[0.0, 1, 0]])
    b1, b2 = torch.tensor([[1.0], [0.0]]), torch.tensor([[0.0], [1.0]])
    result = accumulated_spectrum([(a1, b1), (a2, b2)])
    assert result["stable_rank"] == pytest.approx(2)
    assert result["energy_outside_first_direction"] == pytest.approx(0.5)
    assert accumulated_spectrum([(a1, b1), (a2, b1)])["stable_rank"] == pytest.approx(1)
    s = torch.linalg.svdvals(b1 @ a1 + b2 @ a2)
    torch.testing.assert_close(torch.tensor(result["singular_values"]), s)


def test_math_prefix_unpads_prompt_and_checks_next_token_alignment():
    data = {
        "sequences": torch.tensor([[0, 0, 5, 6, 7, 8, 9, 0, 0]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0, 0]]),
        "response_mask": torch.tensor([[1, 1, 1, 0, 0]]),
    }
    tokens, n = trajectory_prefix(data, response_tokens=2)
    assert tokens.tolist() == [5, 6, 7, 8] and n == 2
    model = tiny_model().eval()
    got = response_log_distribution(model, tokens, n)
    expected = model(tokens[None]).logits[:, 1:3].float().log_softmax(-1)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    zero = distribution_shift(got, got, tokens[-n:])
    assert zero["relora/probe_response_kl_mean"] == 0
    assert zero["relora/probe_chosen_logprob_max_abs_diff"] == 0
    assert zero["relora/probe_argmax_flip_fraction"] == 0


def test_profiles_change_only_merges_and_restart_ramp():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "configs/qwen3-4b-base-grpo-lora-r1-blog.json").read_text())
    profiles = []
    for count in [0, 5]:
        trial = json.loads(
            (root / f"configs/qwen3-4b-base-grpo-relora-r1-warmup{count}.json").read_text()
        )
        cfg = ReLoRATrainConfig.from_cli_overrides(
            [f"{k}={json.dumps(v)}" for k, v in trial.items()]
        )
        validate_relora(cfg)
        assert trial.pop("trainer.relora_merge_interval") == 40
        assert trial.pop("trainer.relora_restart_warmup_updates") == count
        assert trial.pop("trainer.relora_probe_response_tokens") == 128
        assert trial == baseline
        profiles.append(cfg)
    profiles[0].trainer.policy.model.lora.alpha = 1
    with pytest.raises(ValueError, match="standard full-layer"):
        validate_relora(profiles[0])
