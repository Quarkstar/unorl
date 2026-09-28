"""Initialization-only NoRA; ordinary PEFT LoRA thereafter."""

import torch
from peft.tuners.lora.layer import Linear


@torch.no_grad()
def normalize_lora_initialization(model):
    """Normalize each A column over rank, preserving the zero initial update.

    Meta parameters have no values and are left for FSDP to initialize from
    the normalized rank-zero parameters. No forward hooks or constraints remain.
    """
    modules = 0
    entries = 0
    positive = 0
    for layer in model.modules():
        if not isinstance(layer, Linear):
            continue
        for adapter, projection in layer.lora_A.items():
            a = projection.weight
            b = layer.lora_B[adapter].weight
            if not a.is_meta:
                if torch.count_nonzero(b).item():
                    raise ValueError(
                        "NoRA-init requires zero B; cannot normalize a trained adapter"
                    )
                norms = a.float().norm(dim=0, keepdim=True)
                if (norms == 0).any():
                    raise ValueError("NoRA-init requires nonzero A columns")
                a.copy_((a.float() / norms).to(a.dtype))
                positive += int((a > 0).sum().item())
                entries += a.numel()
            modules += 1
    if not modules:
        raise ValueError("NoRA-init found no linear LoRA adapters")
    return {"modules": modules, "materialized_A_entries": entries, "positive_A_entries": positive}
