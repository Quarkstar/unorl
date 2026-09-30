"""Distributed LoRA-FA check using SkyRL's actual FSDP2 strategy.

Run with torchrun on the intended GPU count. Artifacts belong in an ignored
temporary directory, not in the experiment registry.
"""

import argparse
import faulthandler
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from skyrl.train.config import SkyRLTrainConfig
from transformers import Qwen3Config, Qwen3ForCausalLM

from unorl.lorafa_worker import LoRAFAStrategy, LoRAFAWrapper


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
        strategy = LoRAFAStrategy(
            fsdp_config=cfg.trainer.policy.fsdp_config,
            optimizer_config=cfg.trainer.policy.optimizer_config,
            model_config=cfg.trainer.policy.model,
            num_training_steps=3,
        )
        strategy.setup_distributed()
        wrapped = LoRAFAWrapper(
            str(model_path),
            use_flash_attention_2=False,
            bf16=False,
            lora_rank=1,
            lora_alpha=32,
            lora_init_method="lorafa",
            target_modules="all-linear",
            remove_microbatch_padding=False,
            use_torch_compile=False,
            meta_init=dist.get_rank() != 0,
        )
        wrapped.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        wrapped, optimizer, scheduler = strategy.prepare((wrapped, None, None))
        print(f"Rank {dist.get_rank()}: FSDP initialized", flush=True)
        model = wrapped.model
        frozen = {
            n: p.to_local().detach().clone()
            for n, p in model.named_parameters()
            if not p.requires_grad
        }
        tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
        model.train()
        norms = []
        for _ in range(3):
            loss = model(tokens, labels=tokens).loss
            print(f"Rank {dist.get_rank()}: forward finished", flush=True)
            assert torch.isfinite(loss)
            strategy.backward(loss, wrapped, optimizer)
            norms.append(float(strategy.optimizer_step(optimizer, wrapped, scheduler)))
            print(f"Rank {dist.get_rank()}: AdamW update finished", flush=True)
            for name, p in model.named_parameters():
                if name in frozen:
                    torch.testing.assert_close(p.to_local(), frozen[name], rtol=0, atol=0)
                    assert p.grad is None and p not in optimizer.state
        assert all(0 < norm < 100 for norm in norms), norms
        assert len(optimizer.state) == 14
        assert scheduler.last_epoch == 3
        assert isinstance(optimizer, torch.optim.AdamW)
        checkpoint = str(args.output / "checkpoint")
        strategy.save_checkpoint(
            wrapped,
            checkpoint,
            node_local_rank=int(os.environ["LOCAL_RANK"]),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        before_factors = dict(strategy.lorafa_factors)
        strategy.load_checkpoint(wrapped, checkpoint, optimizer=optimizer, scheduler=scheduler)
        assert strategy.lorafa_factors == before_factors
        if dist.get_rank() == 0:
            weights = load_file(
                str(args.output / "checkpoint/lora_adapter/adapter_model.safetensors")
            )
            a_keys = [k for k in weights if "lora_A" in k]
            b_keys = [k for k in weights if "lora_B" in k]
            assert len(a_keys) == len(b_keys) == 14
            assert any(weights[k].abs().sum() > 0 for k in b_keys)
            result = {
                "success": True,
                "world_size": dist.get_world_size(),
                "optimizer": type(optimizer).__name__,
                "updates": 3,
                "corrected_grad_norms": norms,
                "frozen_weights_unchanged": True,
                "optimizer_state_only_B": True,
                "resume_correction_cache_verified": True,
                "native_export_has_A_and_B": True,
            }
            (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
