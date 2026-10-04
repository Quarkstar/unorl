"""Model-side fixed-plane compensation and serialized continuation checks."""

import copy

import pytest
import torch
from test_relora_refresh import warm_model

from unorl.gradual_refresh import apply_plane_refresh
from unorl.low_resource import adapter_layers, effective_weights
from unorl.refresh_transition import accumulate_plane_correction, make_refresh_plane


def saved_planes(model):
    return {
        name: make_refresh_plane(layer.lora_A["default"].weight.detach(), seed=17 + index)
        for index, (name, layer) in enumerate(adapter_layers(model))
    }


def test_increment_preserves_function_parameter_identity_and_selected_adam_state():
    model, optimizer, tokens = warm_model()
    planes = saved_planes(model)
    identities = {name: id(param) for name, param in model.named_parameters()}
    before = {name: value.clone() for name, value in effective_weights(model)}
    logits = model(tokens).logits.detach().clone()
    states = {param: copy.deepcopy(state) for param, state in optimizer.state.items()}
    columns = {}
    cosines = {}
    original_b = {
        name: layer.lora_B["default"].weight.detach().clone()
        for name, layer in adapter_layers(model)
    }

    def capture(name, a, fresh, b, plane, scale):
        columns[name] = accumulate_plane_correction(None, b, a, fresh, plane, scale)
        cosines[name] = ((a * fresh).sum() / (a.norm() * fresh.norm())).clamp(-1, 1)

    metrics = apply_plane_refresh(model, optimizer, planes, 2, on_correction=capture)
    assert metrics["relora/delta_l2"] > 0
    assert set(columns) == set(planes)
    assert identities == {name: id(param) for name, param in model.named_parameters()}
    for name, value in effective_weights(model):
        torch.testing.assert_close(value, before[name], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(model(tokens).logits, logits, atol=1e-6, rtol=1e-4)
    for name, layer in adapter_layers(model):
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        torch.testing.assert_close(b, original_b[name], rtol=0, atol=0)
        for param in (a, b):
            for key in ("step", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizer.state[param][key], states[param][key], rtol=0, atol=0
                )
        torch.testing.assert_close(
            optimizer.state[a]["exp_avg"], states[a]["exp_avg"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            optimizer.state[b]["exp_avg"], states[b]["exp_avg"] * cosines[name], rtol=0, atol=0
        )


def test_serialized_mid_transition_resume_reproduces_model_adam_and_compressed_history(tmp_path):
    model, optimizer, tokens = warm_model()
    planes, columns = saved_planes(model), {}

    def advance(current, opt, basis, history):
        current(tokens, labels=tokens).loss.backward()
        opt.step()
        opt.zero_grad()

        def capture(name, a, fresh, b, plane, scale):
            history[name] = accumulate_plane_correction(
                history.get(name), b, a, fresh, plane, scale
            )

        apply_plane_refresh(current, opt, basis, 2, on_correction=capture)

    for _ in range(3):
        advance(model, optimizer, planes, columns)
    checkpoint = tmp_path / "model-transition.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "planes": planes,
            "columns": columns,
        },
        checkpoint,
    )
    for _ in range(7):
        advance(model, optimizer, planes, columns)
    saved = torch.load(checkpoint, weights_only=True)
    restored, opt, _ = warm_model()
    restored.load_state_dict(saved["model"])
    opt.load_state_dict(saved["optimizer"])
    for _ in range(7):
        advance(restored, opt, saved["planes"], saved["columns"])
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, restored.state_dict()[name], rtol=0, atol=0)
    for actual_state, restored_state in zip(optimizer.state.values(), opt.state.values()):
        for key in actual_state:
            torch.testing.assert_close(actual_state[key], restored_state[key], rtol=0, atol=0)
    for name in columns:
        for actual, replayed in zip(columns[name], saved["columns"][name]):
            torch.testing.assert_close(actual, replayed, rtol=0, atol=0)


def test_zero_increment_and_layer_name_validation():
    model, optimizer, _ = warm_model()
    planes = saved_planes(model)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    apply_plane_refresh(model, optimizer, planes, 0)
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, before_model[name], rtol=0, atol=0)
    for index, state in optimizer.state_dict()["state"].items():
        for key, actual in state.items():
            torch.testing.assert_close(
                actual, before_optimizer["state"][index][key], rtol=0, atol=0
            )
    with pytest.raises(ValueError, match="names must exactly match"):
        apply_plane_refresh(model, optimizer, {}, 2)
