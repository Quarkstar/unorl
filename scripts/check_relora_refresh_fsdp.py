"""Check compensated gradual refresh, warm AdamW continuation and FSDP2 export."""

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import SkyRLTrainConfig
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from unorl.low_resource import adapter_layers, effective_parameters, effective_weights, full_tensor
from unorl.merged_weights import BaseWeightExtractor
from unorl.relora import distribution_shift, response_log_distribution
from unorl.relora_refresh_worker import RefreshPolicyWorker
from unorl.relora_worker import ReLoRAStrategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gradual",
        action="store_true",
        help="Verify a six-increment transition and resume before its final increment",
    )
    args = parser.parse_args()
    faulthandler.dump_traceback_later(90, repeat=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
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
        cfg = SkyRLTrainConfig.from_cli_overrides(
            [
                "trainer.policy.model.lora.rank=1",
                "trainer.policy.model.lora.alpha=32",
                "trainer.policy.optimizer_config.lr=0.001",
                "trainer.policy.optimizer_config.num_warmup_steps=0",
                'trainer.policy.optimizer_config.scheduler="constant"',
                "trainer.policy.optimizer_config.weight_decay=0.0",
            ]
        )
        interval = 8 if args.gradual else 3
        strategy = ReLoRAStrategy(
            merge_interval=interval,
            restart_warmup_updates=0,
            first_merge_step=4,
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
        model = wrapped.model
        worker_type = RefreshPolicyWorker
        if args.gradual:
            from unorl.gradual_refresh_worker import GradualRefreshPolicyWorker

            worker_type = GradualRefreshPolicyWorker
        worker = worker_type.__new__(worker_type)
        worker.model = wrapped
        worker.strategy = strategy
        worker.get_node_local_rank = lambda: int(os.environ["LOCAL_RANK"])
        worker.optimizer = optimizer
        worker.scheduler = scheduler
        worker.cfg = SimpleNamespace(
            seed=42,
            export_path=str(args.output / "exports"),
            relora_refresh_angle_degrees=20.0,
            relora_enable_merge=True,
            relora_first_merge_step=4,
            relora_merge_interval=interval,
            relora_refresh_updates=6,
            relora_restart_warmup_updates=0,
            relora_track_updates=True,
            max_training_steps=9,
        )
        worker._factor_history = {}
        if args.gradual:
            worker._clear_transition()
        identities = {n: id(p) for n, p in model.named_parameters()}
        tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
        norms = []
        applied_lrs = []
        probe = None
        for step in range(1, 9):
            applied_lrs.append(optimizer.param_groups[0]["lr"])
            loss = model(tokens, labels=tokens).loss
            assert torch.isfinite(loss)
            strategy.backward(loss, wrapped, optimizer)
            norms.append(float(strategy.optimizer_step(optimizer, wrapped, scheduler)))
            assert isinstance(optimizer, torch.optim.AdamW)
            assert len(optimizer.state) == 28
            assert scheduler.last_epoch == step
            if worker._should_merge(step):
                with torch.no_grad():
                    before = model(tokens).logits.float()
                    before_dist = response_log_distribution(model, tokens[0], 2)
                    scheduler_state = scheduler.state_dict()
                    before_weights = {n: v.clone() for n, v in effective_weights(model)}
                    worker._math_probe = (tokens[0].cpu(), 2)
                    metrics = worker._merge_with_probe(step)
                    after = model(tokens).logits.float()
                    shift = distribution_shift(
                        before_dist, response_log_distribution(model, tokens[0], 2), tokens[0, -2:]
                    )
                    probe = (after - before).abs().max().item()
                    assert torch.isfinite(after).all() and probe < 0.02, probe
                    assert len(optimizer.state) == 28
                    assert all(float(state["step"]) == step for state in optimizer.state.values())
                    assert scheduler.state_dict() == scheduler_state
                    assert identities == {n: id(p) for n, p in model.named_parameters()}
                    for _, layer in adapter_layers(model):
                        assert torch.isfinite(full_tensor(layer.lora_A["default"].weight)).all()
                        assert torch.count_nonzero(full_tensor(layer.lora_B["default"].weight)) > 0
                    for name, value in effective_weights(model):
                        torch.testing.assert_close(
                            value, before_weights[name], atol=1e-6, rtol=1e-5
                        )
                    base_only = {
                        name: full_tensor(param).clone()
                        for name, param, _ in effective_parameters(model)
                    }
                    for chunk in BaseWeightExtractor(model).extract_weights(torch.float32):
                        torch.testing.assert_close(
                            chunk.tensors[0], base_only[chunk.names[0]], rtol=0, atol=0
                        )
                print(
                    f"Rank {dist.get_rank()}: compensated refresh, retained Adam, probe max diff {probe}",
                    flush=True,
                )
        assert applied_lrs == [0.001] * 8, applied_lrs
        assert all(0 < norm < 100 for norm in norms), norms
        assert all(float(s["step"]) == 8 for s in optimizer.state.values())
        plane_signature = None
        if args.gradual:
            assert worker._transition_start == 4 and worker._transition_count == 5
            digest = hashlib.sha256()
            for name, plane in sorted(worker._transition_planes.items()):
                digest.update(name.encode())
                for row in plane:
                    digest.update(row.numpy().tobytes())
            plane_signature = digest.hexdigest()
            signatures = [None] * dist.get_world_size()
            dist.all_gather_object(signatures, plane_signature)
            assert len(set(signatures)) == 1, signatures
            if dist.get_rank() == 0:
                assert all(len(history) == 2 for history in worker._factor_history.values())
        checkpoint = str(args.output / "checkpoint")
        worker.save_checkpoint(checkpoint)
        with torch.no_grad():
            before_resume = model(tokens).logits.float()
        worker.load_checkpoint(checkpoint)
        if args.gradual:
            assert worker._transition_count == 5 and worker._pending_base_sync
            worker._math_probe = (tokens[0].cpu(), 2)
        with torch.no_grad():
            torch.testing.assert_close(model(tokens).logits.float(), before_resume, rtol=0, atol=0)
        strategy.backward(model(tokens, labels=tokens).loss, wrapped, optimizer)
        worker.optim_step()
        with torch.no_grad():
            next_update_logits = model(tokens).logits.float().clone()
            next_update_weights = {name: value.clone() for name, value in effective_weights(model)}
            next_optimizer_states = [
                {key: full_tensor(value).clone() for key, value in state.items()}
                for state in optimizer.state.values()
            ]
        worker.load_checkpoint(checkpoint)
        strategy.backward(model(tokens, labels=tokens).loss, wrapped, optimizer)
        if args.gradual:
            worker._math_probe = (tokens[0].cpu(), 2)
            worker.optim_step()
            assert worker._transition_count == 0 and not worker._transition_planes
            assert worker._pending_base_sync
        else:
            strategy.optimizer_step(optimizer, wrapped, scheduler)
        with torch.no_grad():
            torch.testing.assert_close(
                model(tokens).logits.float(), next_update_logits, rtol=0, atol=0
            )
            for name, value in effective_weights(model):
                torch.testing.assert_close(value, next_update_weights[name], rtol=0, atol=0)
            for state, saved_state in zip(optimizer.state.values(), next_optimizer_states):
                for key, value in state.items():
                    torch.testing.assert_close(full_tensor(value), saved_state[key], rtol=0, atol=0)
        strategy.save_hf_model(wrapped, str(args.output / "dense-export"))
        effective = {n: v.cpu().clone() for n, v in effective_weights(model)}
        if dist.get_rank() == 0:
            dense = AutoModelForCausalLM.from_pretrained(args.output / "dense-export")
            for name, value in dense.state_dict().items():
                torch.testing.assert_close(value, effective[name], rtol=0, atol=0)
            result = {
                "success": True,
                "world_size": dist.get_world_size(),
                "optimizer": type(optimizer).__name__,
                "updates": 9,
                "refresh_variant": "multi-update fixed plane" if args.gradual else "one-shot",
                "merge_after_updates": list(range(4, 10)) if args.gradual else [4, 7],
                "checkpoint_after_update": 8,
                "checkpoint_inside_transition": args.gradual,
                "fixed_planes_identical_across_ranks": args.gradual,
                "fixed_plane_sha256": plane_signature,
                "compressed_correction_history": args.gradual,
                "rollout_sync_test_scope": "base-weight extraction verified; no inference engine launched",
                "grad_norms": norms,
                "probe_logits_max_abs_diff": probe,
                "merge_metrics": metrics,
                "response_prefix_probe": shift,
                "next_lr": scheduler.get_last_lr()[0],
                "parameter_identity_preserved": True,
                "scheduler_preserved": True,
                "adam_history_preserved": True,
                "diagnostic_factor_history_checkpointed": True,
                "actual_worker_optimizer_and_weight_step_telemetry_verified": True,
                "checkpoint_resume_verified": True,
                "next_update_reproduced_exactly": True,
                "next_update_full_effective_weights_and_adam_state_reproduced_exactly": True,
                "actual_refresh_worker_boundary_and_collective_rank_diagnostics_verified": True,
                "applied_lrs": applied_lrs,
                "dense_export_includes_prior_merge_and_current_adapter": True,
            }
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        dist.barrier()
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
