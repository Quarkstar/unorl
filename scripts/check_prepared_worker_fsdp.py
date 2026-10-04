"""Check real GRPO worker microbatches, preparation, two transfers and resume."""

import argparse
import faulthandler
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from unorl.low_resource import adapter_layers, effective_parameters, effective_weights, full_tensor
from unorl.merged_weights import BaseWeightExtractor
from unorl.prepared_config import PreparedTrainConfig
from unorl.prepared_worker import PreparedPolicyWorker
from unorl.relora_worker import ReLoRAStrategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(120, repeat=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    worker = None
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        model_path = args.output / "tiny-model"
        if dist.get_rank() == 0:
            torch.manual_seed(42)
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
            ).save_pretrained(model_path)
        dist.barrier()
        cfg = PreparedTrainConfig.from_cli_overrides(
            [
                "trainer.policy.model.lora.rank=1",
                "trainer.policy.model.lora.alpha=32",
                "trainer.policy.optimizer_config.lr=0.001",
                "trainer.policy.optimizer_config.num_warmup_steps=0",
                'trainer.policy.optimizer_config.scheduler="constant"',
                "trainer.policy.optimizer_config.weight_decay=0.0",
                "trainer.policy.optimizer_config.max_grad_norm=0.001",
                "trainer.relora_merge_interval=4",
                "trainer.relora_first_merge_step=4",
                "trainer.relora_restart_warmup_updates=0",
                "trainer.prepared_window_updates=2",
                "trainer.prepared_token_chunk=2",
                "trainer.max_training_steps=8",
                "trainer.micro_train_batch_size_per_gpu=2",
                'trainer.algorithm.policy_loss_type="regular"',
                "trainer.algorithm.use_kl_loss=false",
                "trainer.algorithm.use_kl_in_reward=false",
                "trainer.algorithm.grpo_norm_by_std=false",
            ]
        )
        cfg.trainer.export_path = str(args.output / "exports")
        strategy = ReLoRAStrategy(
            merge_interval=4,
            first_merge_step=4,
            restart_warmup_updates=0,
            fsdp_config=cfg.trainer.policy.fsdp_config,
            optimizer_config=cfg.trainer.policy.optimizer_config,
            model_config=cfg.trainer.policy.model,
            num_training_steps=8,
        )
        strategy.setup_distributed()
        wrapped = HFModelWrapper(
            str(model_path),
            use_flash_attention_2=False,
            bf16=False,
            lora_rank=1,
            lora_alpha=32,
            lora_init_method="kaiming",
            target_modules="all-linear",
            remove_microbatch_padding=False,
            use_torch_compile=False,
            meta_init=dist.get_rank() != 0,
        )
        wrapped.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        wrapped, optimizer, scheduler = strategy.prepare((wrapped, None, None))
        worker = PreparedPolicyWorker.__new__(PreparedPolicyWorker)
        worker.model, worker.optimizer, worker.scheduler = wrapped, optimizer, scheduler
        worker.strategy, worker.cfg = strategy, cfg.trainer
        worker.device_mesh = init_device_mesh(
            "cuda", (dist.get_world_size(),), mesh_dim_names=("dp",)
        )
        worker.mesh_rank = SimpleNamespace(dp_size=dist.get_world_size())
        worker.policy_loss_fn = PolicyLossRegistry.get("regular")
        worker.get_node_local_rank = lambda: int(os.environ["LOCAL_RANK"])
        worker._factor_history = {}
        worker._prepared = {}
        worker._prepared_hooks = {}
        worker._prepared_target = worker._prepared_start = None
        worker._previous_delta = worker._before_step_factors = None
        worker._memory_window_active = False
        worker._pending_base_sync = True
        worker._math_probe = None
        worker.merge_metrics = {}
        identities = {name: id(p) for name, p in wrapped.model.named_parameters()}
        lrs, projection_errors = [], []

        def perform(step):
            ids = (
                torch.arange(48, device="cuda").reshape(8, 6) + dist.get_rank() * 3 + step
            ) % 30 + 1
            ids[:, :3] = (torch.arange(3, device="cuda") + dist.get_rank() * 2 + step) % 30 + 1
            attention = torch.ones_like(ids)
            response = torch.ones(8, 3, device="cuda")
            response[::2, 0] = 0  # Unequal numbers of loss-bearing response tokens.
            rewards = torch.tensor([0.0, 1.0] * 4, device="cuda")
            advantages = (rewards - rewards.mean())[:, None] * response
            advantages /= response.sum() * dist.get_world_size()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                old = wrapped(ids, 3, attention_mask=attention).detach()
            batch = TrainingInputBatch(
                {
                    "sequences": ids,
                    "attention_mask": attention,
                    "response_mask": response,
                    "loss_mask": response,
                    "advantages": advantages,
                    "action_log_probs": old,
                },
            )
            batch.metadata = {"response_length": 3}
            lrs.append(optimizer.param_groups[0]["lr"])
            result = worker.forward_backward(batch)
            assert result.metrics and all(
                s.pending_microbatches == 4 for s in worker._prepared.values()
            )
            if step == 3:
                for name, layer in adapter_layers(wrapped.model):
                    state = worker._prepared[name]
                    ga, gb = state.ga.clone(), state.gb.clone()
                    for gradient in (ga, gb):
                        dist.all_reduce(gradient)
                        gradient.div_(dist.get_world_size())
                    a, b = (
                        full_tensor(layer.lora_A["default"].weight),
                        full_tensor(layer.lora_B["default"].weight),
                    )
                    expected = (32 * b.norm() * ga[:1], 32 * a.norm() * gb[:, :1])
                    actual = (
                        full_tensor(layer.lora_A["default"].weight.grad),
                        full_tensor(layer.lora_B["default"].weight.grad),
                    )
                    for reference, observed in zip(expected, actual):
                        projection_errors.append(
                            (reference.double() - observed.double()).norm().item()
                            / max(observed.double().norm().item(), 1e-30)
                        )
            norm = worker.optim_step()
            assert math_isfinite(norm) and scheduler.last_epoch == step
            assert identities == {name: id(p) for name, p in wrapped.model.named_parameters()}

        def snapshot():
            return {
                "weights": {name: t.clone() for name, t in effective_weights(wrapped.model)},
                "optimizer": {
                    name: {k: full_tensor(t).clone() for k, t in optimizer.state[p].items()}
                    for name, p in wrapped.model.named_parameters()
                    if p in optimizer.state
                },
            }

        def assert_snapshot(expected):
            actual = snapshot()
            for name, tensor in actual["weights"].items():
                torch.testing.assert_close(tensor, expected["weights"][name], rtol=0, atol=0)
            for name, states in actual["optimizer"].items():
                for key, tensor in states.items():
                    torch.testing.assert_close(
                        tensor, expected["optimizer"][name][key], rtol=0, atol=0
                    )

        for step in (1, 2, 3):
            perform(step)
        assert worker._prepared_target == 4 and all(s.count == 1 for s in worker._prepared.values())
        worker.save_checkpoint(str(args.output / "active-checkpoint"))
        perform(4)
        perform(5)
        expected = snapshot()
        worker.load_checkpoint(str(args.output / "active-checkpoint"))
        assert worker._prepared_target == 4 and all(s.count == 1 for s in worker._prepared.values())
        perform(4)
        perform(5)
        assert_snapshot(expected)
        assert (
            not worker._prepared
            and min(worker._native_adam_steps().values()) < scheduler.last_epoch
        )
        worker.save_checkpoint(str(args.output / "closed-checkpoint"))
        perform(6)
        expected = snapshot()
        worker.load_checkpoint(str(args.output / "closed-checkpoint"))
        assert not worker._prepared and worker._pending_base_sync
        perform(6)
        assert_snapshot(expected)
        perform(7)
        perform(8)
        assert not worker._prepared and lrs == [0.001] * 11
        assert max(projection_errors) < 0.05, max(projection_errors)
        base_state = {
            name: full_tensor(p).clone() for name, p, _ in effective_parameters(wrapped.model)
        }
        for chunk in BaseWeightExtractor(wrapped.model).extract_weights(torch.float32):
            torch.testing.assert_close(chunk.tensors[0], base_state[chunk.names[0]], rtol=0, atol=0)
        export = args.output / "dense-export"
        strategy.save_hf_model(wrapped, str(export))
        dense = AutoModelForCausalLM.from_pretrained(export, dtype=torch.float32).cuda().eval()
        probe = torch.tensor([[1, 2, 3, 4]], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            source_logits = wrapped.model(probe).logits.float()
            dense_logits = dense(probe).logits.float()
        export_difference = (source_logits - dense_logits).abs().max().item()
        assert export_difference < 0.02, export_difference
        report = {
            "success": True,
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "real_policy_forward_backward": True,
            "real_prepared_optim_step": True,
            "active_and_closed_checkpoint_replay_exact": True,
            "global_scheduler_step": scheduler.last_epoch,
            "native_adam_counter_min": min(worker._native_adam_steps().values()),
            "native_adam_counter_max": max(worker._native_adam_steps().values()),
            "max_relative_native_gradient_projection_error": max(projection_errors),
            "constant_lrs": lrs,
            "parameter_identities_preserved": True,
            "base_weight_extractor_exact": True,
            "dense_export_logit_max_absolute_difference": export_difference,
            "limitations": "Tiny native worker fixture with eight synthetic response trajectories per rank and no live inference engine. Manually bootstrapped model/strategy; not Ray model initialization, real rollout/evaluation, full-scale VRAM or performance evidence.",
        }
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, report)
        if dist.get_rank() == 0:
            (args.output / "result.json").write_text(
                json.dumps({"ranks": reports}, indent=2) + "\n"
            )
            print(json.dumps(report), flush=True)
    finally:
        if worker is not None:
            worker._close_preparation()
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


def math_isfinite(value):
    return value is not None and torch.isfinite(torch.tensor(value)).item()


if __name__ == "__main__":
    main()
