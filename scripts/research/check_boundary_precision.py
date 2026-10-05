"""Compare a saved pre-merge policy with its merged native FSDP policy.

Run with torchrun on the checkpoint's eight GPUs after the shared prefix
exits. No optimizer updates, sampling, checkpoint writes, or live adapters
are added. Fixed held-out texts are used only for numerical diagnostics.
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from torch.distributed.fsdp import FSDPModule
from transformers import AutoTokenizer

from unorl.low_resource import adapter_layers, full_tensor, merge_and_reset
from unorl.merge_precision import (
    adam_step_counters,
    aggregate_output_shifts,
    output_shift,
    weight_rounding_metrics,
)
from unorl.relora_refresh_config import RefreshTrainConfig
from unorl.relora_worker import ReLoRAStrategy

ROOT = Path(__file__).resolve().parents[2]


def fixed_probe(run, step, tokenizer, rank, response_tokens):
    path = run / f"exports/aime25/dumped_evals/global_step_{step}_evals/aime25.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[rank]
    prompt = tokenizer.encode(row["input_prompt"], add_special_tokens=False)
    response = tokenizer.encode(row["output_response"], add_special_tokens=False)[:response_tokens]
    if not prompt or not response or len(prompt) > 2048:
        raise ValueError("Fixed probe requires a valid saved prompt and response")
    ids = torch.tensor(prompt + response, device="cuda")
    return (
        ids,
        len(response),
        {
            "file": str(path.relative_to(ROOT)),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row": rank,
            "prompt_tokens": len(prompt),
            "response_tokens": len(response),
            "token_ids_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
        },
    )


@torch.no_grad()
def response_logits(model, ids, length, control_group):
    logits = (
        model(ids.unsqueeze(0), logits_to_keep=length + 1, use_cache=False).logits[:, :-1].float()
    )
    # Direct HF forwards bypass the native worker's inference resharding.
    # Restore FP32 sharded parameters before inspecting or merging them,
    # with all ranks kept in the same audit phase.
    torch.cuda.synchronize()
    dist.barrier(group=control_group)
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
    dist.barrier(group=control_group)
    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--response-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", choices=("training", "sdpa"), default="training")
    args = parser.parse_args()
    if args.step != 40 or not 1 <= args.response_tokens <= 256:
        parser.error("This shared-prefix experiment uses step 40 and at most 256 probe tokens")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    control_group = dist.new_group(backend="gloo")
    try:
        run = ROOT / "runs" / args.run_id
        raw_config = json.loads((run / "config.json").read_text())
        if raw_config["trainer.relora_enable_merge"]:
            raise ValueError("Expected the unmerged shared-prefix checkpoint")
        cfg = RefreshTrainConfig.from_cli_overrides(
            [f"{key}={json.dumps(value)}" for key, value in raw_config.items()]
        ).trainer
        if dist.get_world_size() != cfg.placement.policy_num_gpus_per_node:
            raise ValueError("Audit world size must equal the checkpoint world size")
        checkpoint = run / f"checkpoints/global_step_{args.step}"
        trainer = torch.load(
            checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        if trainer["global_step"] != args.step:
            raise ValueError("Wrong origin checkpoint")
        strategy = ReLoRAStrategy(
            merge_interval=cfg.relora_merge_interval,
            restart_warmup_updates=0,
            first_merge_step=cfg.relora_first_merge_step,
            fsdp_config=cfg.policy.fsdp_config,
            optimizer_config=cfg.policy.optimizer_config,
            model_config=cfg.policy.model,
            seed=cfg.seed,
            micro_train_batch_size_per_gpu=cfg.micro_train_batch_size_per_gpu,
            num_training_steps=cfg.max_training_steps,
        )
        strategy.setup_distributed()
        wrapped = HFModelWrapper(
            cfg.policy.model.path,
            use_flash_attention_2=cfg.flash_attn if args.attention_backend == "training" else False,
            bf16=False,
            lora_rank=1,
            lora_alpha=32,
            lora_dropout=0,
            lora_init_method="kaiming",
            target_modules="all-linear",
            remove_microbatch_padding=False,
            use_torch_compile=False,
            meta_init=dist.get_rank() != 0,
        )
        wrapped, optimizer, scheduler = strategy.prepare((wrapped, None, None))
        _, restored = strategy.load_checkpoint(
            wrapped, str(checkpoint / "policy"), optimizer=optimizer, scheduler=scheduler
        )
        counters = adam_step_counters(optimizer.state.values())
        if (
            scheduler.last_epoch != args.step
            or not counters
            or min(counters) != args.step
            or max(counters) != args.step
            or any(restored.get("client_state", {}).get("relora_factor_history", {}).values())
        ):
            raise ValueError("Expected standard-LoRA state at the first pre-merge boundary")
        model = wrapped.model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.policy.model.path, local_files_only=True)
        ids, length, probe_source = fixed_probe(
            run, args.step, tokenizer, dist.get_rank(), args.response_tokens
        )
        torch.cuda.reset_peak_memory_stats()
        before = response_logits(model, ids, length, control_group)
        repeat = response_logits(model, ids, length, control_group)
        baseline = output_shift(before, repeat, ids[-length:])
        del repeat
        layers = {}
        for name, layer in adapter_layers(model):
            base = full_tensor(layer.get_base_layer().weight)
            a = full_tensor(layer.lora_A["default"].weight)
            b = full_tensor(layer.lora_B["default"].weight)
            if dist.get_rank() == 0:
                layers[name] = weight_rounding_metrics(base, a, b, layer.scaling["default"])
        dist.barrier(group=control_group)
        merge_metrics = merge_and_reset(model, seed=cfg.seed + args.step * 10000)
        optimizer.state.clear()
        after = response_logits(model, ids, length, control_group)
        shift = output_shift(before, after, ids[-length:])
        row = {
            "rank": dist.get_rank(),
            "probe_source": probe_source,
            "baseline_repeat": baseline,
            "merge": shift,
            "scheduler_step": scheduler.last_epoch,
            "restored_adam_step_min": min(counters),
            "restored_adam_step_max": max(counters),
            "optimizer_entries_after_reset": len(optimizer.state),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, row, group=control_group)
        if dist.get_rank() == 0:
            totals = {
                key: sum(layer[key] for layer in layers.values())
                for key in next(iter(layers.values()))
            }
            report = {
                "run_id": args.run_id,
                "checkpoint_global_step": args.step,
                "attention_backend": args.attention_backend,
                "config_sha256": hashlib.sha256((run / "config.json").read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "baseline_repeat": aggregate_output_shifts([r["baseline_repeat"] for r in rows]),
                "merge_shift": aggregate_output_shifts([r["merge"] for r in rows]),
                "weight_totals": totals,
                "layers": layers,
                "ranks": rows,
                "merge_metrics": merge_metrics,
                "limitations": "Native FSDP mixed-precision forwards on fixed saved response prefixes, with the attention backend recorded explicitly. An SDPA diagnostic is not a FlashAttention or vLLM kernel check. No accuracy claim, full-response coverage, or optimizer-continuity test. Probe examples are not used to choose adapter directions. Source checkpoint is read-only.",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output)
            print(
                json.dumps(
                    {
                        "baseline_repeat": report["baseline_repeat"],
                        "merge_shift": report["merge_shift"],
                    }
                ),
                flush=True,
            )
    except Exception:
        # A failed rank must exit promptly so torchrun can stop its peers.
        # Destroying NCCL here can hide the original exception while another
        # rank is waiting in an unmatched model-forward collective.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
