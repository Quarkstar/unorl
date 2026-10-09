"""Measure BF16 base-compensation representation error from native shards.

CPU-only, streams dense linear weights one layer at a time. Does not run
the policy, measure actual matmul error, adapter casting, or accuracy.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]


def energy(tensor):
    return tensor.double().square().sum().item()


def fingerprint(tensor):
    return hashlib.sha256(tensor.float().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = ROOT / "runs" / args.run_id
    config = json.loads((run / "config.json").read_text())
    model_root = Path(config["trainer.policy.model.path"])
    index_path = model_root / "model.safetensors.index.json"
    index_raw = index_path.read_bytes()
    index = json.loads(index_raw)["weight_map"]
    policy = run / f"checkpoints/global_step_{args.step}/policy"
    extra_path = policy / "extra_state_world_size_8_rank_0.pt"
    extra_raw = extra_path.read_bytes()
    extra = torch.load(extra_path, map_location="cpu", weights_only=False)
    if extra["lr_scheduler"]["last_epoch"] != args.step:
        raise ValueError("Checkpoint step differs from request")
    history = extra["client_state"]["relora_factor_history"]
    if not history or config["trainer.policy.model.lora.rank"] != 1:
        raise ValueError("Expected a merged rank-one run with correction history")
    shard_paths = [policy / f"model_world_size_8_rank_{rank}.pt" for rank in range(8)]
    metadata = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size} for path in shard_paths}
    shards = [
        torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in shard_paths
    ]
    layers = {}
    for number, (name, pairs) in enumerate(sorted(history.items()), start=1):
        key = name + ".base_layer.weight"
        locals_ = []
        for shard in shards:
            value = shard[key]
            if (
                value.dtype != torch.float32
                or len(value.placements) != 1
                or not value.placements[0].is_shard(0)
            ):
                raise ValueError("Expected one-dimensional FP32 row-sharded native bases")
            locals_.append(value.to_local())
        merged = torch.cat(locals_, dim=0)
        original_key = name.removeprefix("base_model.model.") + ".weight"
        with safe_open(model_root / index[original_key], framework="pt", device="cpu") as handle:
            original_bf16 = handle.get_tensor(original_key)
        if original_bf16.dtype != torch.bfloat16 or merged.shape != original_bf16.shape:
            raise ValueError("Native and original BF16 base tensors differ in dtype or shape")
        original = original_bf16.float()
        delta = merged - original
        cast_delta = merged.to(torch.bfloat16).float() - original
        frozen = sum(col.float() @ row.float() for row, col in pairs)
        changed = delta != 0
        layers[name] = {
            "elements": merged.numel(),
            "fp32_changed_elements": changed.sum().item(),
            "changed_elements_lost_in_bf16": ((cast_delta == 0) & changed).sum().item(),
            "base_energy": energy(original),
            "fp32_compensation_energy": energy(delta),
            "bf16_compensation_energy": energy(cast_delta),
            "base_cast_residual_energy": energy(cast_delta - delta),
            "storage_vs_factor_compensation_error_energy": energy(delta - frozen),
            "compensation_inner_product": (delta.double() * cast_delta.double()).sum().item(),
            "native_base_fp32_tensor_sha256": fingerprint(merged),
            "original_base_fp32_tensor_sha256": fingerprint(original),
            "original_model_key": original_key,
            "original_shard": index[original_key],
            "correction_pairs": len(pairs),
        }
        if number % 36 == 0:
            print(f"Audited {number}/{len(history)} base matrices", flush=True)
    totals = {
        key: sum(row[key] for row in layers.values())
        for key in next(iter(layers.values()))
        if key.endswith("energy") or key.endswith("elements") or key == "compensation_inner_product"
    }
    compensation = totals["fp32_compensation_energy"]
    if compensation <= 0:
        raise ValueError("Expected nonzero saved base compensation")
    changed_elements = sum(row["fp32_changed_elements"] for row in layers.values())
    lost_elements = sum(row["changed_elements_lost_in_bf16"] for row in layers.values())
    summary = {
        "layers": len(layers),
        "fp32_compensation_l2": compensation**0.5,
        "bf16_compensation_l2": totals["bf16_compensation_energy"] ** 0.5,
        "base_cast_residual_l2": totals["base_cast_residual_energy"] ** 0.5,
        "relative_base_cast_error": (totals["base_cast_residual_energy"] / compensation) ** 0.5,
        "relative_base_cast_error_to_base": (
            totals["base_cast_residual_energy"] / totals["base_energy"]
        )
        ** 0.5,
        "storage_vs_factor_relative_error": (
            totals["storage_vs_factor_compensation_error_energy"] / compensation
        )
        ** 0.5,
        "compensation_cast_cosine": totals["compensation_inner_product"]
        / (compensation * totals["bf16_compensation_energy"]) ** 0.5,
        "fraction_changed_elements_lost_in_bf16": lost_elements / changed_elements,
    }
    report = {
        "run_id": args.run_id,
        "step": args.step,
        "summary": summary,
        "totals": totals,
        "layers": layers,
        "native_shard_sources": metadata,
        "extra_state_sha256": hashlib.sha256(extra_raw).hexdigest(),
        "model_index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": "Actual native FP32 bases versus original BF16 linear weights, reconstructed by concatenating validated Shard(0) local tensors. Each used tensor is fingerprinted in normalized FP32 bytes; whole weight-shard files are not hashed. Isolates BF16 base casting of cumulative saved compensation, excluding adapter casting, actual matmul/activation errors, receiver tensor equality, output KL, reward causality, and accuracy. Not a full-model rank measurement; live training unchanged.",
    }
    output = ROOT / "research/data" / f"{args.run_id}-base-cast-step{args.step}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
