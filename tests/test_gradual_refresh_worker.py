"""Unsharded worker schedule, checkpoint state and multi-increment replay."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_relora_refresh import warm_model

from unorl.gradual_refresh_worker import GradualRefreshPolicyWorker


def make_worker(model, optimizer, root):
    root.mkdir(parents=True, exist_ok=True)
    worker = GradualRefreshPolicyWorker.__new__(GradualRefreshPolicyWorker)
    worker.model = SimpleNamespace(model=model)
    worker.optimizer = optimizer
    worker.scheduler = SimpleNamespace(last_epoch=4)
    worker.cfg = SimpleNamespace(
        seed=42,
        export_path=str(root / "exports"),
        relora_first_merge_step=5,
        relora_merge_interval=20,
        relora_refresh_updates=10,
        relora_refresh_angle_degrees=20,
        relora_enable_merge=True,
    )
    worker._factor_history = {}
    worker._before_step_factors = None
    worker._previous_delta = None
    worker._clear_transition()
    return worker


def test_worker_schedule_and_serialized_mid_transition_replay(tmp_path, monkeypatch):
    monkeypatch.setattr("unorl.gradual_refresh_worker.dist.get_rank", lambda: 0)
    monkeypatch.setattr(
        "unorl.gradual_refresh_worker.dist.all_reduce", lambda *args, **kwargs: None
    )
    model, optimizer, tokens = warm_model()
    worker = make_worker(model, optimizer, tmp_path / "original")
    assert [step for step in range(1, 46) if worker._should_merge(step)] == list(
        range(5, 15)
    ) + list(range(25, 35)) + [45]

    def advance(current, step):
        current.model.model(tokens, labels=tokens).loss.backward()
        current.optimizer.step()
        current.optimizer.zero_grad()
        current.scheduler.last_epoch = step
        current._math_probe = (tokens[0], 2)
        result = current._merge_with_probe(step)
        assert result["refresh/transition_increment"] == step - 4
        assert all(len(history) == 2 for history in current._factor_history.values())
        return result

    for step in range(5, 8):
        advance(worker, step)
    checkpoint = tmp_path / "worker-state.pt"

    def save_strategy_checkpoint(**kwargs):
        torch.save(
            {
                "model": kwargs["model"].model.state_dict(),
                "optimizer": kwargs["optimizer"].state_dict(),
                "client_state": kwargs["client_state"],
            },
            kwargs["ckpt_dir"],
        )

    worker.strategy = SimpleNamespace(save_checkpoint=save_strategy_checkpoint)
    worker.get_node_local_rank = lambda: 0
    worker.save_checkpoint(checkpoint)
    for step in range(8, 15):
        advance(worker, step)
    assert worker._transition_state()["count"] == 0
    assert worker._transition_state()["planes"] == {}
    saved = torch.load(checkpoint, weights_only=True)
    restored_model, restored_optimizer, _ = warm_model()
    restored_model.load_state_dict(saved["model"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored = make_worker(restored_model, restored_optimizer, tmp_path / "replayed")

    def load_strategy_checkpoint(**kwargs):
        kwargs["model"].model.load_state_dict(saved["model"])
        kwargs["optimizer"].load_state_dict(saved["optimizer"])
        kwargs["scheduler"].last_epoch = 7
        return None, {"client_state": saved["client_state"]}

    restored.strategy = SimpleNamespace(load_checkpoint=load_strategy_checkpoint)
    restored._pending_base_sync = False
    restored.load_checkpoint(checkpoint)
    assert restored._pending_base_sync
    for step in range(8, 15):
        advance(restored, step)
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, restored_model.state_dict()[name], rtol=0, atol=0)
    for left, right in zip(optimizer.state.values(), restored_optimizer.state.values()):
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    for name, actual in worker._factor_history.items():
        for left, right in zip(actual, restored._factor_history[name]):
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_reject_corrupt_transition_state(tmp_path, monkeypatch):
    monkeypatch.setattr("unorl.gradual_refresh_worker.dist.get_rank", lambda: 0)
    model, optimizer, _ = warm_model()
    worker = make_worker(model, optimizer, tmp_path)
    worker._begin_transition(5)
    worker._transition_count = 1
    state = copy.deepcopy(worker._transition_state())
    worker._restore_transition(state, 5)
    for field, value, message in (
        ("count", 2, "increment count"),
        ("start", 4, "Transition start"),
        ("planes", {}, "plane names"),
        ("schedule", {}, "schedule differs"),
    ):
        damaged = copy.deepcopy(state)
        damaged[field] = value
        with pytest.raises(ValueError, match=message):
            worker._restore_transition(damaged, 5)
    damaged = copy.deepcopy(state)
    name = next(iter(damaged["planes"]))
    damaged["planes"][name][0].fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        worker._restore_transition(damaged, 5)
    with pytest.raises(ValueError, match="increment count"):
        worker._restore_transition(state, 15)


def test_matched_config_rejects_overlapping_or_truncated_transitions():
    from unorl.gradual_refresh_config import GradualRefreshTrainConfig, validate_gradual_refresh

    configs = Path(__file__).resolve().parents[1] / "configs"
    path = configs / "qwen3-4b-base-grpo-gradual-refresh-r1.json"
    profile = json.loads(path.read_text())
    reference = json.loads((configs / "qwen3-4b-base-grpo-relora-refresh-r1.json").read_text())
    assert profile["trainer.relora_refresh_updates"] == 10
    assert {
        key: value for key, value in profile.items() if key != "trainer.relora_refresh_updates"
    } == reference
    cfg = GradualRefreshTrainConfig.from_cli_overrides(
        [f"{key}={json.dumps(value)}" for key, value in profile.items()]
    )
    validate_gradual_refresh(cfg)
    worker = GradualRefreshPolicyWorker.__new__(GradualRefreshPolicyWorker)
    worker.cfg = cfg.trainer
    assert [step for step in range(1, 101) if worker._should_merge(step)] == list(
        range(40, 50)
    ) + list(range(80, 90))
    for invalid in (0, 41, 1.5, True):
        cfg.trainer.relora_refresh_updates = invalid
        with pytest.raises(ValueError, match="Refresh updates"):
            validate_gradual_refresh(cfg)
    cfg.trainer.relora_refresh_updates = 10
    cfg.trainer.max_training_steps = 85
    with pytest.raises(ValueError, match="finish partway"):
        validate_gradual_refresh(cfg)
