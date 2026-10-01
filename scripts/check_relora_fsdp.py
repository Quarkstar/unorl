"""Check ReLoRA restart/merge, continued AdamW training and export on actual FSDP2."""

import argparse
import faulthandler
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import SkyRLTrainConfig
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from unorl.low_resource import adapter_layers, effective_weights, full_tensor, merge_and_reset
from unorl.merged_weights import BaseWeightExtractor
from unorl.relora import distribution_shift, response_log_distribution
from unorl.relora_worker import ReLoRAStrategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
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
        strategy = ReLoRAStrategy(
            merge_interval=3,
            restart_warmup_updates=2,
            fsdp_config=cfg.trainer.policy.fsdp_config,
            optimizer_config=cfg.trainer.policy.optimizer_config,
            model_config=cfg.trainer.policy.model,
            num_training_steps=7,
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
        identities = {n: id(p) for n, p in model.named_parameters()}
        tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
        norms = []
        applied_lrs = []
        probe = None
        for step in range(1, 8):
            applied_lrs.append(optimizer.param_groups[0]["lr"])
            loss = model(tokens, labels=tokens).loss
            assert torch.isfinite(loss)
            strategy.backward(loss, wrapped, optimizer)
            norms.append(float(strategy.optimizer_step(optimizer, wrapped, scheduler)))
            assert isinstance(optimizer, torch.optim.AdamW)
            assert len(optimizer.state) == 28
            assert scheduler.last_epoch == step
            if step in (3, 6):
                with torch.no_grad():
                    before = model(tokens).logits.float()
                    before_dist = response_log_distribution(model, tokens[0], 2)
                    scheduler_state = scheduler.state_dict()
                    before_weights = {n: v.clone() for n, v in effective_weights(model)}
                    metrics = merge_and_reset(model, seed=12345 + step, init_method="kaiming")
                    optimizer.state.clear()
                    after = model(tokens).logits.float()
                    shift = distribution_shift(
                        before_dist, response_log_distribution(model, tokens[0], 2), tokens[0, -2:]
                    )
                    probe = (after - before).abs().max().item()
                    assert torch.isfinite(after).all() and probe < 0.02, probe
                    assert not optimizer.state
                    assert scheduler.state_dict() == scheduler_state
                    assert identities == {n: id(p) for n, p in model.named_parameters()}
                    for _, layer in adapter_layers(model):
                        assert torch.isfinite(full_tensor(layer.lora_A["default"].weight)).all()
                        assert torch.count_nonzero(full_tensor(layer.lora_B["default"].weight)) == 0
                    for chunk in BaseWeightExtractor(model).extract_weights(torch.float32):
                        torch.testing.assert_close(chunk.tensors[0], before_weights[chunk.names[0]])
                print(
                    f"Rank {dist.get_rank()}: merged, reset Adam, probe max diff {probe}",
                    flush=True,
                )
        assert applied_lrs == [0.001, 0.001, 0.001, 0.0, 0.001, 0.001, 0.0], applied_lrs
        assert all(0 < norm < 100 for norm in norms), norms
        assert any(float(s["step"]) == 1 for s in optimizer.state.values())
        checkpoint = str(args.output / "checkpoint")
        strategy.save_checkpoint(
            wrapped,
            checkpoint,
            node_local_rank=int(os.environ["LOCAL_RANK"]),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        with torch.no_grad():
            before_resume = model(tokens).logits.float()
        strategy.load_checkpoint(wrapped, checkpoint, optimizer=optimizer, scheduler=scheduler)
        with torch.no_grad():
            torch.testing.assert_close(model(tokens).logits.float(), before_resume, rtol=0, atol=0)
        strategy.backward(model(tokens, labels=tokens).loss, wrapped, optimizer)
        strategy.optimizer_step(optimizer, wrapped, scheduler)
        with torch.no_grad():
            next_update_logits = model(tokens).logits.float().clone()
        strategy.load_checkpoint(wrapped, checkpoint, optimizer=optimizer, scheduler=scheduler)
        strategy.backward(model(tokens, labels=tokens).loss, wrapped, optimizer)
        strategy.optimizer_step(optimizer, wrapped, scheduler)
        with torch.no_grad():
            torch.testing.assert_close(
                model(tokens).logits.float(), next_update_logits, rtol=0, atol=0
            )
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
                "updates": 7,
                "merge_after_updates": [3, 6],
                "grad_norms": norms,
                "probe_logits_max_abs_diff": probe,
                "merge_metrics": metrics,
                "response_prefix_probe": shift,
                "next_lr": scheduler.get_last_lr()[0],
                "parameter_identity_preserved": True,
                "scheduler_preserved": True,
                "adam_history_reset_and_rebuilt": True,
                "checkpoint_resume_verified": True,
                "next_update_reproduced_exactly": True,
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
