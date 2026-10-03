"""First-principles checks for compensated refresh and optimizer continuity."""

import copy
import math

import pytest
import torch
from test_loft import tiny_model

from unorl.low_resource import adapter_layers, effective_weights
from unorl.relora import accumulated_spectrum
from unorl.relora_refresh import refresh_adapter, rotated_row


def test_rotation_preserves_norm_angle_and_effective_weight_gradient():
    torch.manual_seed(42)
    dtype = torch.float64
    a = torch.randn(1, 19, dtype=dtype, requires_grad=True)
    b = torch.randn(13, 1, dtype=dtype, requires_grad=True)
    base = torch.randn(13, 19, dtype=dtype)
    gradient = torch.randn_like(base)
    scale = 32
    old_weight = base + scale * b @ a
    ga, gb = torch.autograd.grad((old_weight * gradient).sum(), (a, b))
    new_a = rotated_row(a.detach(), 20, torch.Generator().manual_seed(17)).requires_grad_()
    correction = scale * b.detach() @ (a.detach() - new_a.detach())
    new_weight = base + correction + scale * b @ new_a
    new_ga, new_gb = torch.autograd.grad((new_weight * gradient).sum(), (new_a, b))
    torch.testing.assert_close(new_weight, old_weight)
    torch.testing.assert_close(new_ga, ga)
    cosine = math.cos(math.radians(20))
    sine = math.sin(math.radians(20))
    orthogonal = (new_a.detach() - cosine * a.detach()) / sine
    torch.testing.assert_close(
        (orthogonal * a).sum(), torch.tensor(0.0, dtype=dtype), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(torch.linalg.vector_norm(new_a), torch.linalg.vector_norm(a))
    torch.testing.assert_close(new_gb, cosine * gb + sine * scale * gradient @ orthogonal.T)
    assert torch.linalg.vector_norm(new_gb - gb) <= scale * torch.linalg.vector_norm(
        gradient
    ) * torch.linalg.vector_norm(new_a - a)
    for angle in [-1, 90]:
        with pytest.raises(ValueError):
            rotated_row(a, angle, torch.Generator())


def warm_model():
    model = tiny_model().eval()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    for _ in range(4):
        model(tokens, labels=tokens).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model, optimizer, tokens


def test_refresh_preserves_warm_b_state_counters_and_creates_future_rank_capacity():
    model, optimizer, tokens = warm_model()
    identities = {n: id(p) for n, p in model.named_parameters()}
    before_weights = {n: p.clone() for n, p in effective_weights(model)}
    before_logits = model(tokens).logits.detach().clone()
    states = {p: copy.deepcopy(state) for p, state in optimizer.state.items()}
    before_b = {
        name: layer.lora_B["default"].weight.detach().clone()
        for name, layer in adapter_layers(model)
    }
    corrections = {}
    metrics = refresh_adapter(
        model,
        optimizer,
        seed=123,
        angle_degrees=20,
        on_correction=lambda name, a, b: corrections.setdefault(name, (a.clone(), b.clone())),
    )
    assert metrics["relora/delta_l2"] > 0
    assert identities == {n: id(p) for n, p in model.named_parameters()}
    for name, value in effective_weights(model):
        torch.testing.assert_close(value, before_weights[name], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(model(tokens).logits, before_logits, atol=1e-6, rtol=1e-4)
    for name, layer in adapter_layers(model):
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        torch.testing.assert_close(b, before_b[name], rtol=0, atol=0)
        for param in (a, b):
            torch.testing.assert_close(
                optimizer.state[param]["step"], states[param]["step"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                optimizer.state[param]["exp_avg_sq"], states[param]["exp_avg_sq"], rtol=0, atol=0
            )
        torch.testing.assert_close(
            optimizer.state[a]["exp_avg"], states[a]["exp_avg"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            optimizer.state[b]["exp_avg"], states[b]["exp_avg"] * math.cos(math.radians(20))
        )
        result = accumulated_spectrum(
            [corrections[name], (a.detach(), b.detach() * layer.scaling["default"])]
        )
        assert result["energy_outside_first_direction"] < 1e-5
    for _ in range(3):
        model(tokens, labels=tokens).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    ranks = [
        accumulated_spectrum(
            [
                corrections[name],
                (
                    layer.lora_A["default"].weight.detach(),
                    layer.lora_B["default"].weight.detach() * layer.scaling["default"],
                ),
            ]
        )["energy_outside_first_direction"]
        for name, layer in adapter_layers(model)
    ]
    assert max(ranks) > 1e-4
    assert all(s["step"] == 7 for s in optimizer.state.values())


def test_zero_rotation_preserves_the_standard_optimizer_next_update_exactly():
    model, optimizer, tokens = warm_model()
    other = copy.deepcopy(model)
    replay = torch.optim.AdamW(
        [p for p in other.parameters() if p.requires_grad], lr=0.001, weight_decay=0
    )
    replay.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    refresh_adapter(other, replay, seed=99, angle_degrees=0)
    for current, opt in [(model, optimizer), (other, replay)]:
        current(tokens, labels=tokens).loss.backward()
        opt.step()
        opt.zero_grad()
    torch.testing.assert_close(model(tokens).logits, other(tokens).logits, rtol=0, atol=0)


def test_checkpoint_branch_profiles_match_and_boundary_phase_is_correct():
    import json
    from pathlib import Path

    from unorl.relora_refresh_config import RefreshTrainConfig, validate_refresh
    from unorl.relora_refresh_worker import RefreshPolicyWorker

    root = Path(__file__).resolve().parents[1]
    profiles = []
    for name in ("standard", "relora-refresh"):
        profile = json.loads(
            (root / f"configs/qwen3-4b-base-grpo-{name}-r1-continue.json").read_text()
        )
        cfg = RefreshTrainConfig.from_cli_overrides(
            [f"{k}={json.dumps(v)}" for k, v in profile.items()]
        )
        validate_refresh(cfg)
        worker = RefreshPolicyWorker.__new__(RefreshPolicyWorker)
        worker.cfg = cfg.trainer
        expected = [101, 141, 181] if name == "relora-refresh" else []
        assert [s for s in range(100, 201) if worker._should_merge(s)] == expected
        assert profile["trainer.resume_path"].endswith("global_step_100")
        assert profile["trainer.max_training_steps"] == 200
        profile.pop("trainer.relora_enable_merge")
        profile.pop("trainer.relora_refresh_angle_degrees")
        profiles.append(profile)
    assert profiles[0] == profiles[1]


def test_base_model_refresh_profiles_match_original_budget_and_recipe():
    import json
    from pathlib import Path

    from unorl.relora_refresh_config import RefreshTrainConfig, validate_refresh
    from unorl.relora_refresh_worker import RefreshPolicyWorker

    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "configs/qwen3-4b-base-grpo-lora-r1-blog.json").read_text())
    profiles = []
    for name in ("relora-refresh-r1", "standard-r1-refresh-control"):
        profile = json.loads((root / f"configs/qwen3-4b-base-grpo-{name}.json").read_text())
        cfg = RefreshTrainConfig.from_cli_overrides(
            [f"{k}={json.dumps(v)}" for k, v in profile.items()]
        )
        validate_refresh(cfg)
        assert cfg.trainer.max_training_steps == 100
        assert cfg.trainer.resume_mode is None
        assert "trainer.resume_path" not in profile
        worker = RefreshPolicyWorker.__new__(RefreshPolicyWorker)
        worker.cfg = cfg.trainer
        expected = [40, 80] if name == "relora-refresh-r1" else []
        assert [s for s in range(1, 101) if worker._should_merge(s)] == expected
        for key, value in reference.items():
            if key != "trainer.ckpt_interval":
                assert profile[key] == value, key
        profile.pop("trainer.relora_enable_merge")
        profile.pop("trainer.relora_refresh_angle_degrees")
        profiles.append(profile)
    assert profiles[0] == profiles[1]


def test_weight_space_update_telemetry_matches_dense_norm_and_direction():
    from unorl.relora_refresh import factor_inner, update_factors

    torch.manual_seed(23)
    a, b = torch.randn(1, 11, dtype=torch.float64), torch.randn(7, 1, dtype=torch.float64)
    new_a, new_b = a + 0.03 * torch.randn_like(a), b + 0.03 * torch.randn_like(b)
    later_a, later_b = new_a + 0.02 * torch.randn_like(a), new_b + 0.02 * torch.randn_like(b)
    first = update_factors(a, b, new_a, new_b, 32)
    second = update_factors(new_a, new_b, later_a, later_b, 32)
    dense1 = 32 * (new_b @ new_a - b @ a)
    dense2 = 32 * (later_b @ later_a - new_b @ new_a)
    assert factor_inner(first, first) == pytest.approx(dense1.square().sum().item(), rel=1e-12)
    assert factor_inner(first, second) == pytest.approx((dense1 * dense2).sum().item(), rel=1e-12)
