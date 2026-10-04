"""Isolate compensation rounding with saved real adapters; never train or use a GPU."""

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from unorl.refresh_transition import make_refresh_plane, rotate_in_refresh_plane

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def simulate(base, a, b, plane, increments, dtype, scale):
    """Hold B and gradients fixed to isolate rounding, not learning performance."""
    weight = base.to(dtype)
    start = weight.double()
    initial_a = a.clone()
    current = a.clone()
    energies, errors = [], []
    for _ in range(increments):
        fresh = rotate_in_refresh_plane(current, plane, 20 / increments)
        correction = (b * scale) @ (current - fresh)
        before = weight.float()
        weight = (before + correction).to(dtype)
        energies.append(correction.double().square().sum().item())
        errors.append(
            (weight.double() - before.double() - correction.double()).square().sum().item()
        )
        current = fresh
    exact_correction = (b.double() * scale) @ (initial_a.double() - current.double())
    endpoint_error = weight.double() - start - exact_correction
    target_l2 = exact_correction.norm().item()
    return (
        {
            "stored_base_dtype": str(dtype),
            "increments": increments,
            "target_endpoint_correction_l2": target_l2,
            "endpoint_effective_weight_error_l2": endpoint_error.norm().item(),
            "endpoint_relative_correction_error": endpoint_error.norm().item() / target_l2,
            "sum_increment_error_l2": sum(math.sqrt(value) for value in errors),
            "increment_rms_relative_rounding": math.sqrt(sum(errors) / sum(energies)),
            "unchanged_base_element_fraction": (weight.double() == start).double().mean().item(),
        },
        weight,
        current,
    )


def main():
    torch.set_num_threads(2)
    adapter_path = (
        ROOT
        / "runs/qwen3-4b-base-grpo-lora-r1-blog-20260923-01/exports/global_step_100/policy/adapter_model.safetensors"
    )
    model = ROOT / "models/Qwen3-4B-Base"
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    selected = ["model.layers.0.self_attn.q_proj", "model.layers.35.self_attn.v_proj"]
    layers = {}
    with safe_open(str(adapter_path), framework="pt", device="cpu") as adapters:
        for number, name in enumerate(selected):
            a = adapters.get_tensor(f"base_model.model.{name}.lora_A.weight").float()
            b = adapters.get_tensor(f"base_model.model.{name}.lora_B.weight").float()
            key = f"{name}.weight"
            with safe_open(str(model / index[key]), framework="pt", device="cpu") as weights:
                base = weights.get_tensor(key).float()
            plane = make_refresh_plane(a, seed=42 + number)
            records = {}
            for dtype in (torch.float32, torch.bfloat16):
                one, one_weight, one_a = simulate(base, a, b, plane, 1, dtype, 32)
                ten, ten_weight, ten_a = simulate(base, a, b, plane, 10, dtype, 32)
                assert torch.allclose(one_a, ten_a, atol=1e-7, rtol=1e-5)
                records[str(dtype)] = {
                    "one_shot": one,
                    "ten_increments": ten,
                    "endpoint_base_difference_l2": (one_weight.double() - ten_weight.double())
                    .norm()
                    .item(),
                    "endpoint_bf16_base_element_disagreement_fraction": (
                        one_weight.bfloat16() != ten_weight.bfloat16()
                    )
                    .double()
                    .mean()
                    .item(),
                    "endpoint_a_difference_l2": (one_a.double() - ten_a.double()).norm().item(),
                }
            assert (
                records["torch.float32"]["ten_increments"]["endpoint_relative_correction_error"]
                < 1e-3
            )
            layers[name] = {
                "shape": list(base.shape),
                "loaded_base_fp32_sha256": hashlib.sha256(
                    base.contiguous().numpy().tobytes()
                ).hexdigest(),
                "source_shard": index[key],
                "results": records,
            }
    report = {
        "source_adapter": str(adapter_path.relative_to(ROOT)),
        "source_adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "source_model_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "scale": 32,
        "layers": layers,
        "limitations": "Two real projection matrices and historical step-100 adapter only. CPU diagnostic holds B fixed and performs no optimizer updates; no rollout, gradient, logits, or task accuracy is measured. FP32 stored base matches current SkyRL training initialization; BF16 stored base is a counterfactual. Forward casts and separate LoRA matmuls can introduce additional errors not captured by effective-weight algebra. This does not reproduce actual step-40/80 trajectories or establish performance equivalence.",
    }
    output = ROOT / "research/data/refresh-rounding-analysis.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: row["results"] for name, row in layers.items()}))


if __name__ == "__main__":
    main()
