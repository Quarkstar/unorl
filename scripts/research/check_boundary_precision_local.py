"""Check a verified native pre-merge checkpoint with one BF16 inference replica.

Reconstruct each native tensor from its eight CPU shards and verify that every
frozen tensor matches the original backbone and every adapter tensor matches
the saved PEFT export. Merge with the native FP32 adapter factors, then cast
the resulting weight to BF16 as in native mixed-precision forwarding. This
avoids distributed forwards; it does not validate FSDP or vLLM execution.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from unorl.low_resource import adapter_layers
from unorl.merge_precision import (
    adam_step_counters,
    aggregate_output_shifts,
    output_shift,
    weight_rounding_metrics,
)

ROOT = Path(__file__).resolve().parents[2]


def native_tensor(shards, key):
    """Materialize one native tensor on CPU, without distributed collectives."""
    values = [shard[key] for shard in shards]
    if len(values[0].placements) != 1:
        raise ValueError("Expected the checkpoint's one-dimensional FSDP mesh")
    placement = values[0].placements[0]
    if any(value.placements != values[0].placements for value in values):
        raise ValueError("Inconsistent native tensor placements")
    if placement.is_shard(0):
        return torch.cat([value.to_local() for value in values], dim=0)[: values[0].shape[0]]
    if placement.is_replicate():
        result = values[0].to_local()
        if not all(torch.equal(result, value.to_local()) for value in values[1:]):
            raise ValueError("Inconsistent replicated native tensor")
        return result
    raise ValueError("Unsupported native checkpoint placement")


def verify_checkpoint(run, base_path, adapter, adapter_path):
    policy = run / "checkpoints/global_step_40/policy"
    index = json.loads((base_path / "model.safetensors.index.json").read_text())["weight_map"]
    base_config = json.loads((base_path / "config.json").read_text())
    aliases = {}
    shards = [
        torch.load(
            policy / f"model_world_size_8_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for rank in range(8)
    ]
    frozen = adapters = 0
    for key in shards[0]:
        value = native_tensor(shards, key)
        if ".lora_A." in key or ".lora_B." in key:
            export_key = key.replace(".default.", ".")
            if not torch.equal(value, adapter[export_key]):
                raise ValueError(f"Native/exported adapter mismatch: {key}")
            adapters += 1
        else:
            original_key = key.removeprefix("base_model.model.").replace(".base_layer.", ".")
            if original_key == "lm_head.weight" and original_key not in index:
                if not base_config.get("tie_word_embeddings", False):
                    raise ValueError("Missing untied output head in the original backbone")
                aliases[original_key] = "model.embed_tokens.weight"
                original_key = aliases[original_key]
            with safe_open(base_path / index[original_key], framework="pt", device="cpu") as handle:
                original = handle.get_tensor(original_key)
            if not torch.equal(value, original.float()):
                raise ValueError(f"Frozen native/original backbone mismatch: {key}")
            frozen += 1
    ranks = []
    for rank in range(8):
        extra = torch.load(
            policy / f"extra_state_world_size_8_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer = torch.load(
            policy / f"optim_world_size_8_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        counters = adam_step_counters(optimizer["state"].values())
        if (
            extra["lr_scheduler"]["last_epoch"] != 40
            or not counters
            or min(counters) != 40
            or max(counters) != 40
            or any(extra["client_state"].get("relora_factor_history", {}).values())
        ):
            raise ValueError("Expected the shared, unmerged step-40 optimizer state")
        ranks.append({"rank": rank, "adam_states": len(counters), "adam_step": 40})
    return {
        "verified_frozen_tensors": frozen,
        "verified_adapter_tensors": adapters,
        "verified_tied_weight_aliases": aliases,
        "adapter_export_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "ranks": ranks,
    }


@torch.no_grad()
def forwards(model, probes):
    outputs = []
    for ids, length, _ in probes:
        logits = (
            model(ids.unsqueeze(0), logits_to_keep=length + 1, use_cache=False)
            .logits[:, :-1]
            .float()
        )
        outputs.append(logits.cpu())
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--attention-backend", choices=("flash_attention_2", "sdpa"), default="flash_attention_2"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = ROOT / "runs" / args.run_id
    cfg = json.loads((run / "config.json").read_text())
    checkpoint = run / "checkpoints/global_step_40"
    if cfg["trainer.relora_enable_merge"] or cfg["trainer.policy.model.lora.rank"] != 1:
        raise ValueError("Expected the standard rank-one shared prefix")
    trainer = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
    if trainer["global_step"] != 40:
        raise ValueError("Wrong shared checkpoint")
    base_path = Path(cfg["trainer.policy.model.path"])
    adapter_dir = checkpoint / "policy/lora_adapter"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    adapter = load_file(adapter_path)
    verification = verify_checkpoint(run, base_path, adapter, adapter_path)
    print(json.dumps(verification), flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attention_backend,
        device_map="cuda:0",
        local_files_only=True,
    )
    model = (
        PeftModel.from_pretrained(
            base, adapter_dir, is_trainable=False, autocast_adapter_dtype=False
        )
        .to(dtype=torch.bfloat16)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    probe_file = run / "exports/aime25/dumped_evals/global_step_40_evals/aime25.jsonl"
    examples = [json.loads(line) for line in probe_file.read_text().splitlines()]
    probes = []
    # One response from each of eight questions; avoid eight repeats of the
    # first question. Evaluation texts only measure numerical output shifts.
    seen = set()
    for index, row in enumerate(examples):
        if row["input_prompt"] in seen:
            continue
        prompt = tokenizer.encode(row["input_prompt"], add_special_tokens=False)
        response = tokenizer.encode(row["output_response"], add_special_tokens=False)[:128]
        if not prompt or not response or len(prompt) > 2048:
            continue
        seen.add(row["input_prompt"])
        ids = torch.tensor(prompt + response, device="cuda")
        probes.append((ids, len(response), {"row": index, "prompt_tokens": len(prompt)}))
        if len(probes) == 8:
            break
    if len(probes) != 8:
        raise ValueError("Need eight distinct valid fixed numerical probes")
    torch.cuda.reset_peak_memory_stats()
    before = forwards(model, probes)
    repeat = forwards(model, probes)
    baseline = [output_shift(x, y, p[0][-p[1] :].cpu()) for x, y, p in zip(before, repeat, probes)]
    del repeat
    layers = {}
    with torch.no_grad():
        for count, (name, layer) in enumerate(adapter_layers(model), start=1):
            a = adapter[name + ".lora_A.weight"].to("cuda")
            b = adapter[name + ".lora_B.weight"].to("cuda")
            weight = layer.get_base_layer().weight
            original = weight.float()
            scale = layer.scaling["default"]
            layers[name] = weight_rounding_metrics(original, a, b, scale)
            # Reproduce FP32 native merge followed by BF16 forward casting.
            weight.copy_((original + (b @ a) * scale).to(torch.bfloat16))
            layer.lora_B["default"].weight.zero_()
            # A is irrelevant when B is zero; keeping it avoids any RNG use.
            if count % 36 == 0:
                print(f"Merged numerical replica: {count}/252 layers", flush=True)
    after = forwards(model, probes)
    shifts = [output_shift(x, y, p[0][-p[1] :].cpu()) for x, y, p in zip(before, after, probes)]
    totals = {
        key: sum(layer[key] for layer in layers.values()) for key in next(iter(layers.values()))
    }
    report = {
        "run_id": args.run_id,
        "checkpoint_global_step": 40,
        "execution": "single-GPU BF16 inference replica; native FP32 merge factors",
        "attention_backend": args.attention_backend,
        "verification": verification,
        "baseline_repeat": aggregate_output_shifts(baseline),
        "merge_shift": aggregate_output_shifts(shifts),
        "weight_totals": totals,
        "layers": layers,
        "probes": [dict(p[2], response_tokens=p[1]) for p in probes],
        "probe_file_sha256": hashlib.sha256(probe_file.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "limitations": "All native frozen tensors and exported adapter factors verified exactly. Independent BF16 forward replica, not native FSDP or vLLM execution. Fixed short prefixes do not establish reward causality or full-response equivalence. No checkpoint mutation, optimizer update, sampling, or use of held-out examples for direction selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"baseline_repeat": report["baseline_repeat"], "merge_shift": report["merge_shift"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
