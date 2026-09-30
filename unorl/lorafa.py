"""Rank-one LoRA-FA with frozen A and inverse-Gram gradient correction."""

import math

import torch
from peft.tuners.lora.layer import Linear
from torch.distributed.tensor import DTensor


def freeze_lora_a(model):
    """Freeze A before FSDP and reject unsupported adapter configurations."""
    modules = 0
    for layer in model.modules():
        if not isinstance(layer, Linear):
            continue
        if list(layer.lora_A) != ["default"] or list(layer.lora_B) != ["default"]:
            raise ValueError("LoRA-FA supports exactly one default adapter")
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        if a.shape[0] != 1 or b.shape[1] != 1:
            raise ValueError("This LoRA-FA implementation requires rank one")
        if layer.use_dora.get("default", False):
            raise ValueError("LoRA-FA does not support DoRA")
        a.requires_grad_(False)
        if not b.requires_grad:
            raise ValueError("LoRA-FA requires trainable B")
        modules += 1
    if not modules:
        raise ValueError("LoRA-FA found no linear adapters")
    if any(p.requires_grad and ".lora_B." not in n for n, p in model.named_parameters()):
        raise ValueError("LoRA-FA requires all non-B parameters to be frozen")
    return modules


@torch.no_grad()
def rank_one_correction(a, scale, regularization=1e-8):
    """PEFT's regularized inverse Gram is a scalar at rank one.

    Gather only the small A row once after FSDP initialization or resume.
    Its value stays fixed throughout training; no dense weight is constructed.
    """
    if a.shape[0] != 1 or a.requires_grad:
        raise ValueError("Correction requires frozen rank-one A")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("LoRA-FA requires finite positive scaling")
    if regularization < 0 or not math.isfinite(regularization):
        raise ValueError("Gram regularization must be finite and nonnegative")
    value = a.full_tensor() if isinstance(a, DTensor) else a
    norm_sq = value.float().square().sum().item()
    if not math.isfinite(norm_sq) or norm_sq <= 0:
        raise ValueError("LoRA-FA requires finite nonzero A")
    return 1.0 / (scale * scale * (norm_sq + regularization))


def correction_factors(model):
    """Cache one scalar per B, with the same names as FSDP2 parameters."""
    factors = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, Linear):
            continue
        key = f"{name}.lora_B.default.weight" if name else "lora_B.default.weight"
        factors[key] = rank_one_correction(layer.lora_A["default"].weight, layer.scaling["default"])
    if not factors:
        raise ValueError("LoRA-FA found no correction factors")
    return factors


@torch.no_grad()
def correct_gradients(model, factors):
    """Correct accumulated (possibly sharded) B gradients before clipping."""
    corrected = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name not in factors or not parameter.requires_grad:
            raise ValueError(f"Unexpected LoRA-FA gradient: {name}")
        parameter.grad.mul_(factors[name])
        corrected += 1
    return corrected
