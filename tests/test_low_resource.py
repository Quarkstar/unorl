"""CPU checks for signed gradients, merge/reset, dense exports, and configuration."""

import copy
import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from unorl.low_resource import (
    ConstantLearningRate,
    adapter_layers,
    effective_weights,
    make_sgd,
    merge_and_reset,
    reinforce_loss,
    signed_returns,
)
from unorl.low_resource_config import LowResourceTrainConfig, validate_low_resource
from unorl.low_resource_train import batch_mean_advantage, batch_normalized_advantage
from unorl.low_resource_worker import MergedWeightExtractor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=[False, True], ids=["untied", "tied"])
def model(request):
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=request.param,
        attention_dropout=0.0,
    )
    model = get_peft_model(
        Qwen3ForCausalLM(config), LoraConfig(r=1, lora_alpha=1, target_modules="all-linear")
    )
    with torch.no_grad():
        for _, layer in adapter_layers(model):
            layer.lora_B["default"].weight.normal_(std=0.01)
    assert not model.get_input_embeddings().weight.requires_grad
    assert not model.get_output_embeddings().weight.requires_grad
    return model.eval()


def test_signed_gradient_and_padding():
    logits = torch.zeros(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    advantages = signed_returns(rewards, mask) / 2
    log_probs = logits.log_softmax(-1)[..., 0]
    loss, _ = reinforce_loss(log_probs, None, advantages, None, mask)
    expected = -(log_probs[0].sum() - log_probs[1, :2].sum()) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (logits.grad[0, :, 0] < 0).all()
    assert (logits.grad[1, :2, 0] > 0).all()
    assert (logits.grad[1, 2] == 0).all()


@pytest.mark.parametrize("score", [0.0, float("nan"), float("inf"), 0.5, -2.0])
def test_reject_nonbinary_reward(score):
    with pytest.raises(ValueError, match="terminal rewards"):
        signed_returns(torch.tensor([[score]]), torch.ones(1, 1))


def test_all_failed_batch_keeps_negative_signal():
    result = signed_returns(-torch.ones(4, 1), torch.ones(4, 1))
    assert (result == -1).all()


def test_batch_mean_advantage_centers_without_std_scaling():
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    advantages, returns = batch_mean_advantage(rewards, mask)
    expected = torch.tensor([[2 / 3, 2 / 3, 2 / 3], [2 / 3, 2 / 3, 2 / 3], [-4 / 3, -4 / 3, 0.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_batch_normalized_advantage_uses_population_standard_deviation():
    rewards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    advantages, returns = batch_normalized_advantage(rewards, mask)
    expected = torch.tensor(
        [
            [1 / (2**0.5), 1 / (2**0.5), 1 / (2**0.5)],
            [1 / (2**0.5), 1 / (2**0.5), 1 / (2**0.5)],
            [-(2**0.5), -(2**0.5), 0.0],
        ]
    )
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)
    torch.testing.assert_close(advantages[:, 0].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(advantages[:, 0].std(unbiased=False), torch.tensor(1.0))


def test_batch_normalized_advantage_is_zero_for_zero_variance_batch():
    rewards = torch.tensor([[-1.0, 0.0, 0.0]] * 4)
    mask = torch.ones_like(rewards)
    advantages, _ = batch_normalized_advantage(rewards, mask)
    assert torch.count_nonzero(advantages) == 0


def test_reinforce_advantages_use_batch_token_mean():
    from unorl.low_resource_train import ReinforceTrainer

    trainer = ReinforceTrainer.__new__(ReinforceTrainer)
    data = {
        "advantages": torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, 0.0]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
    }
    result = trainer._normalize_advantages(data, [(0, 2)])
    expected = torch.tensor([[0.2, 0.2, 0.2], [-0.2, -0.2, 0.0]])
    torch.testing.assert_close(result["advantages"], expected)


def test_merge_preserves_outputs_and_parameter_identity(model):
    tokens = torch.tensor([[1, 4, 8, 2]])
    before = model(tokens).logits.detach()
    identities = {name: id(value) for name, value in model.named_parameters()}
    old_a = [layer.lora_A["default"].weight.clone() for _, layer in adapter_layers(model)]
    metrics = merge_and_reset(model, seed=123)
    torch.testing.assert_close(before, model(tokens).logits, atol=2e-6, rtol=2e-5)
    assert identities == {name: id(value) for name, value in model.named_parameters()}
    assert metrics["relora/merged_layers"] == 14
    assert metrics["relora/rounding_relative_l2"] < 1e-4
    for previous, (_, layer) in zip(old_a, adapter_layers(model), strict=True):
        assert not torch.equal(previous, layer.lora_A["default"].weight)
        assert not layer.lora_B["default"].weight.count_nonzero()
        assert not layer.get_base_layer().weight.requires_grad


def test_dense_sync_contains_previous_merges_and_active_adapter(model):
    merge_and_reset(model, seed=12)
    with torch.no_grad():
        for _, layer in adapter_layers(model):
            layer.lora_B["default"].weight.normal_(std=0.02)
    dense = Qwen3ForCausalLM(copy.deepcopy(model.config)).eval()
    dense.load_state_dict(dict(effective_weights(model)), strict=True)
    tokens = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model(tokens).logits, dense(tokens).logits, atol=2e-6, rtol=2e-5)
    extractor = MergedWeightExtractor(model)
    metadata = extractor.get_weight_metadata(torch.float32)
    chunks = list(extractor.extract_weights(torch.float32))
    assert metadata["names"] == [chunk.names[0] for chunk in chunks]
    assert metadata["shapes"] == [chunk.shapes[0] for chunk in chunks]
    assert set(metadata["names"]) == set(dense.state_dict())


def test_sgd_has_no_state_across_merges_and_resume(model, tmp_path):
    optimizer = make_sgd(model.parameters(), lr=0.01)
    scheduler = ConstantLearningRate(optimizer)
    tokens = torch.tensor([[1, 2, 3]])
    frozen_before = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
    model(tokens).logits.square().sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    assert not optimizer.state
    assert not scheduler.state_dict()
    assert scheduler.get_last_lr() == [0.01]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            torch.testing.assert_close(p, frozen_before[name], rtol=0, atol=0)
    merge_and_reset(model, seed=3)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "scheduler": scheduler.state_dict()}, checkpoint)
    restored = copy.deepcopy(model)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["model"])
    torch.testing.assert_close(model(tokens).logits, restored(tokens).logits)
    model(tokens).logits.square().sum().backward()
    optimizer.step()
    assert not optimizer.state
    assert any(layer.lora_B["default"].weight.count_nonzero() for _, layer in adapter_layers(model))


def test_profile_and_fail_closed_validation():
    profile = json.loads((ROOT / "configs/qwen3-4b-base-reinforce-lora-r1-blog.json").read_text())
    cfg = LowResourceTrainConfig.from_cli_overrides(
        [f"{key}={json.dumps(value)}" for key, value in profile.items()]
    )
    validate_low_resource(cfg)
    assert cfg.trainer.low_resource.lora_rank == 1
    assert cfg.generator.n_samples_per_prompt == 1
    assert cfg.trainer.algorithm.loss_reduction == "token_mean"
    assert cfg.trainer.train_batch_size == 256
    assert cfg.trainer.policy_mini_batch_size == 256
    assert cfg.trainer.epochs == 4
    cfg.trainer.update_epochs_per_batch = 2
    with pytest.raises(ValueError, match="one update"):
        validate_low_resource(cfg)


def test_batch_mean_profile_validates():
    profile = json.loads((ROOT / "configs/qwen3-4b-base-reinforce-batchmean-r1.json").read_text())
    cfg = LowResourceTrainConfig.from_cli_overrides(
        [f"{key}={json.dumps(value)}" for key, value in profile.items()]
    )
    validate_low_resource(cfg)
    assert cfg.trainer.algorithm.advantage_estimator == "batch_mean_reinforce"


def test_batch_normalized_profile_validates():
    profile = json.loads((ROOT / "configs/qwen3-4b-base-reinforce-batchnorm-r1.json").read_text())
    cfg = LowResourceTrainConfig.from_cli_overrides(
        [f"{key}={json.dumps(value)}" for key, value in profile.items()]
    )
    validate_low_resource(cfg)
    assert cfg.trainer.algorithm.advantage_estimator == "batch_norm_reinforce"


def _distributed_merge_worker(rank, init_file):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        torch.manual_seed(4)
        base = torch.nn.Sequential(torch.nn.Linear(4, 6, bias=False))
        model = get_peft_model(base, LoraConfig(r=1, lora_alpha=1, target_modules=["0"]))
        for _, layer in adapter_layers(model):
            with torch.no_grad():
                layer.lora_B["default"].weight.fill_(0.1)
        expected = {name: value.clone() for name, value in effective_weights(model)}
        mesh = init_device_mesh("cpu", (2,))
        for name, parameter in list(model.named_parameters()):
            module_name, parameter_name = name.rsplit(".", 1)
            value = distribute_tensor(parameter.detach(), mesh, [Shard(0)])
            setattr(
                model.get_submodule(module_name),
                parameter_name,
                torch.nn.Parameter(value, requires_grad=parameter.requires_grad),
            )
        merge_and_reset(model, seed=24)
        for name, value in effective_weights(model):
            torch.testing.assert_close(value, expected[name])
        for _, layer in adapter_layers(model):
            assert not layer.lora_B["default"].weight.full_tensor().count_nonzero()
            a = layer.lora_A["default"].weight.full_tensor()
            copies = [torch.empty_like(a) for _ in range(2)]
            dist.all_gather(copies, a)
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_sharded_rank_one_merge_with_empty_shards(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_merge_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True
    )


def test_skyrl_checkpoint_export_and_merge_schedule(model, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch.distributed as dist
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    from skyrl.train.config.config import FSDPConfig, ModelConfig, OptimizerConfig

    from unorl.low_resource_worker import LowResourcePolicyWorker, SGDStrategy

    # Exercise real checkpoint IO on CPU; only CUDA runtime calls are stubbed.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'checkpoint-group'}", rank=0, world_size=1
    )
    try:
        strategy = SGDStrategy(
            fsdp_config=FSDPConfig(), optimizer_config=OptimizerConfig(), model_config=ModelConfig()
        )
        strategy.world_size = 1
        wrapped = HFModelWrapper.__new__(HFModelWrapper)
        torch.nn.Module.__init__(wrapped)
        wrapped.model = model
        optimizer = make_sgd(model.parameters(), lr=0.01)
        scheduler = ConstantLearningRate(optimizer)
        worker = LowResourcePolicyWorker.__new__(LowResourcePolicyWorker)
        worker.strategy, worker.model = strategy, wrapped
        worker.optimizer, worker.scheduler = optimizer, scheduler
        worker.cfg = SimpleNamespace(seed=42, low_resource=SimpleNamespace(merge_interval=2))
        tokens = torch.tensor([[1, 2, 3]])

        def update():
            model(tokens).logits.square().sum().backward()
            worker.optim_step()

        update()
        assert not worker.merge_metrics
        update()
        assert worker.merge_metrics["relora/merged_layers"] == 14
        expected = model(tokens).logits.detach().clone()
        saved_lr = optimizer.param_groups[0]["lr"]
        path = str(tmp_path / "global_step_2")
        strategy.save_checkpoint(
            model=wrapped,
            ckpt_dir=path,
            node_local_rank=0,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        assert torch.load(Path(path) / "optim_world_size_1_rank_0.pt", weights_only=True) == {}
        extra = torch.load(Path(path) / "extra_state_world_size_1_rank_0.pt", weights_only=False)
        assert extra["lr_scheduler"] == {}
        update()
        strategy.load_checkpoint(
            model=wrapped, ckpt_dir=path, optimizer=optimizer, scheduler=scheduler
        )
        assert scheduler.last_epoch == 2
        assert optimizer.param_groups[0]["lr"] == saved_lr
        assert not optimizer.state
        torch.testing.assert_close(model(tokens).logits, expected, rtol=0, atol=0)
        update()
        assert not worker.merge_metrics
        update()
        assert worker.merge_metrics["relora/merged_layers"] == 14
        export = str(tmp_path / "merged-hf")
        strategy.save_hf_model(wrapped, export)
        dense = Qwen3ForCausalLM.from_pretrained(export).eval()
        torch.testing.assert_close(model(tokens).logits, dense(tokens).logits, atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


def test_skyrl_accepts_registered_algorithm():
    from skyrl.train.utils.utils import validate_cfg

    from unorl.low_resource_train import register_algorithms

    profile = json.loads((ROOT / "configs/qwen3-4b-base-reinforce-lora-r1-blog.json").read_text())
    cfg = LowResourceTrainConfig.from_cli_overrides(
        [f"{key}={json.dumps(value)}" for key, value in profile.items()]
    )
    register_algorithms()
    validate_cfg(cfg)
    validate_low_resource(cfg)
