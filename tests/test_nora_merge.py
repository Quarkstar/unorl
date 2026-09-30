"""Merge/reset must retain learned weights, retire Adam history, and sync both parts."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import torch
from test_nora import tiny_model

from unorl.low_resource import adapter_layers, effective_weights
from unorl.merged_weights import BaseWeightExtractor, MergedWeightExtractor
from unorl.nora import normalize_lora_initialization
from unorl.nora_merge_config import NoRAMergeTrainConfig, validate_nora_merge
from unorl.nora_merge_worker import NoRAMergePolicyWorker, reset_nora_adapter
from unorl.nora_worker import NoRAInitPolicyWorker


def test_merge_preserves_policy_and_parameter_identity_but_retires_adam():
    model = tiny_model(1).eval()
    normalize_lora_initialization(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tokens = torch.tensor([[1, 2, 3, 4]])
    for _ in range(2):
        model(tokens, labels=tokens).loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    before = model(tokens).logits.detach()
    identities = {name: id(p) for name, p in model.named_parameters()}
    old_a = {
        name: layer.lora_A["default"].weight.detach().clone()
        for name, layer in adapter_layers(model)
    }
    state = scheduler.state_dict()
    metrics = reset_nora_adapter(model, optimizer, seed=100)
    assert metrics["relora/retired_optimizer_state_entries"] == 28
    assert not optimizer.state
    assert scheduler.state_dict() == state
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    torch.testing.assert_close(model(tokens).logits, before, rtol=1e-4, atol=1e-6)
    for name, layer in adapter_layers(model):
        assert (layer.lora_A["default"].weight.abs() == 1).all()
        assert not torch.equal(layer.lora_A["default"].weight, old_a[name])
        assert torch.count_nonzero(layer.lora_B["default"].weight) == 0
        assert not layer.get_base_layer().weight.requires_grad
    model(tokens, labels=tokens).loss.backward()
    optimizer.step()
    assert len(optimizer.state) == 28
    assert any(p["step"] == 1 for p in optimizer.state.values())


def test_base_sync_excludes_active_adapter_but_retains_prior_merges():
    model = tiny_model(1)
    normalize_lora_initialization(model)
    with torch.no_grad():
        for _, layer in adapter_layers(model):
            layer.lora_B["default"].weight.fill_(0.001)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    accumulated = dict(effective_weights(model))
    reset_nora_adapter(model, optimizer, seed=101)
    with torch.no_grad():
        for _, layer in adapter_layers(model):
            layer.lora_B["default"].weight.fill_(0.002)
    base_extractor = BaseWeightExtractor(model)
    base = {c.names[0]: c.tensors[0] for c in base_extractor.extract_weights(torch.float32)}
    merged = {
        c.names[0]: c.tensors[0]
        for c in MergedWeightExtractor(model).extract_weights(torch.float32)
    }
    assert set(base) == set(accumulated) == set(merged)
    assert set(base) == set(base_extractor.get_weight_metadata(torch.float32)["names"])
    for name in base:
        torch.testing.assert_close(base[name], accumulated[name])
    assert any(not torch.equal(base[name], merged[name]) for name in base)
    assert not any("lora_" in name or "base_layer" in name for name in base)


def test_sync_base_then_adapter_only_when_needed_and_retry_on_failure(monkeypatch):
    monkeypatch.setattr("unorl.nora_merge_worker.dist.get_rank", lambda: 1)
    worker = object.__new__(NoRAMergePolicyWorker)
    worker._pending_base_sync = True
    worker._is_lora = True
    modes = []

    async def native(*args, **kwargs):
        modes.append(worker._is_lora)

    with patch.object(NoRAInitPolicyWorker, "broadcast_to_inference_engines", new=native):
        asyncio.run(worker.broadcast_to_inference_engines(None, None))
        asyncio.run(worker.broadcast_to_inference_engines(None, None))
        worker._pending_base_sync = True
        asyncio.run(worker.broadcast_to_inference_engines(None, None))
    assert modes == [False, True, True, False, True]
    assert not worker._pending_base_sync and worker._is_lora
    worker._pending_base_sync = True
    with patch.object(
        NoRAInitPolicyWorker,
        "broadcast_to_inference_engines",
        new=AsyncMock(side_effect=RuntimeError("transport failed")),
    ):
        with pytest.raises(RuntimeError, match="transport failed"):
            asyncio.run(worker.broadcast_to_inference_engines(None, None))
    assert worker._pending_base_sync and worker._is_lora


def test_merge_profile_changes_only_interval_and_keeps_native_lora_inference():
    root = Path(__file__).resolve().parents[1]
    trial = json.loads((root / "configs/qwen3-4b-base-grpo-nora-merge-r1.json").read_text())
    reference = json.loads((root / "configs/qwen3-4b-base-grpo-lora-r1-nora-init.json").read_text())
    assert trial.pop("trainer.nora_merge_interval") == 40
    assert trial == reference
    cfg = NoRAMergeTrainConfig.from_cli_overrides(
        [
            f"{k}={json.dumps(v)}"
            for k, v in json.loads(
                (root / "configs/qwen3-4b-base-grpo-nora-merge-r1.json").read_text()
            ).items()
        ]
    )
    validate_nora_merge(cfg)
    assert cfg.trainer.policy.model.lora.rank == 1
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 1
    with pytest.raises(ValueError, match="constant LR without warmup"):
        validate_nora_merge(cfg)


def test_reset_rejects_sgd_before_mutating_model():
    model = tiny_model(1)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="native AdamW"):
        reset_nora_adapter(model, optimizer, seed=42)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
