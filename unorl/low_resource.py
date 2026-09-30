"""Stateless REINFORCE and rank-one adapter operations, independent of Ray."""

import math
from collections.abc import Iterator

import torch
from peft.tuners.lora import LoraLayer
from torch.distributed.tensor import DTensor, distribute_tensor


def signed_returns(rewards: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Broadcast a terminal +1/-1 outcome without centering or whitening."""
    scores = rewards.sum(dim=-1, keepdim=True)
    if not torch.isfinite(scores).all() or not ((scores == 1) | (scores == -1)).all():
        raise ValueError("REINFORCE requires exactly +1/-1 terminal rewards")
    return scores * response_mask.to(rewards.dtype)


def reinforce_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None
):
    """Sum signed log likelihoods; the trainer supplies batch-mean scaling."""
    terms = -log_probs * advantages.detach()
    if loss_mask is not None:
        terms = torch.where(loss_mask.bool(), terms * loss_mask, 0.0)
    return terms.sum(), {"clip_ratio": 0.0}


def make_sgd(parameters, lr: float) -> torch.optim.SGD:
    """Only adapters are trainable; no momentum, decay, or optimizer buffers."""
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable adapter parameters")
    return torch.optim.SGD(trainable, lr=lr, momentum=0.0, weight_decay=0.0, foreach=False)


class ConstantLearningRate:
    """SkyRL scheduler interface with a fixed LR and no saved scheduler state."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0

    def step(self) -> None:
        self.last_epoch += 1

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError("ConstantLearningRate expects no saved scheduler state")


def full_tensor(value: torch.Tensor) -> torch.Tensor:
    """Gather one parameter, never a whole model, on the mesh device."""
    if isinstance(value, DTensor):
        return value.to(value.device_mesh.device_type).full_tensor().detach()
    return value.detach()


@torch.no_grad()
def copy_parameter(parameter: torch.Tensor, value: torch.Tensor) -> None:
    """Write back shards without replacing Parameter objects used by SGD/FSDP."""
    if isinstance(parameter, DTensor):
        value = distribute_tensor(value.contiguous(), parameter.device_mesh, parameter.placements)
    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def adapter_layers(model) -> Iterator[tuple[str, LoraLayer]]:
    for name, layer in model.named_modules():
        if isinstance(layer, LoraLayer):
            if list(layer.lora_A) != ["default"] or layer.r["default"] != 1:
                raise ValueError("Only one rank-one default adapter is supported")
            if layer.merged or layer.fan_in_fan_out or layer.use_dora.get("default", False):
                raise ValueError("Expected unmerged, ordinary linear LoRA")
            yield name, layer


def adapter_delta(layer: LoraLayer) -> torch.Tensor:
    a = full_tensor(layer.lora_A["default"].weight).float()
    b = full_tensor(layer.lora_B["default"].weight).float()
    return (b @ a) * layer.scaling["default"]


@torch.no_grad()
def merge_and_reset(model, seed: int, init_method: str = "kaiming") -> dict[str, float]:
    """Accumulate BA into W and restart A/random, B/zero on every rank.

    FP32 base storage is the default to retain small repeated updates. The
    rounding metric exposes precision lost when users select BF16 base storage.
    """
    if init_method not in {"kaiming", "nora_init"}:
        raise ValueError("Merge/reset supports Kaiming or NoRA-init")
    delta_energy = 0.0
    error_energy = 0.0
    count = 0
    for count, (_, layer) in enumerate(adapter_layers(model), start=1):
        weight = layer.get_base_layer().weight
        base = full_tensor(weight).float()
        delta = adapter_delta(layer)
        merged = (base + delta).to(weight.dtype)
        if not torch.isfinite(merged).all():
            raise FloatingPointError("Nonfinite merged LoRA weights")
        delta_energy += delta.square().sum().item()
        error_energy += (merged.float() - base - delta).square().sum().item()
        copy_parameter(weight, merged)
        a = layer.lora_A["default"].weight
        b = layer.lora_B["default"].weight
        # CPU generator gives every rank identical full factors, including when
        # rank-one tensors have empty shards on some ranks.
        generator = torch.Generator(device="cpu").manual_seed(seed + count)
        fresh_a = torch.empty(tuple(a.shape), dtype=torch.float32, device="cpu")
        torch.nn.init.kaiming_uniform_(fresh_a, a=math.sqrt(5), generator=generator)
        if init_method == "nora_init":
            fresh_a.sign_()
            if not (fresh_a.abs() == 1).all():
                raise ValueError("NoRA-init requires nonzero fresh A entries")
        copy_parameter(a, fresh_a.to(base.device))
        copy_parameter(b, torch.zeros(tuple(b.shape), dtype=b.dtype, device=base.device))
    if not count:
        raise ValueError("No rank-one LoRA layers to merge")
    return {
        "relora/merged_layers": float(count),
        "relora/delta_l2": math.sqrt(delta_energy),
        "relora/rounding_relative_l2": math.sqrt(error_energy / max(delta_energy, 1e-30)),
    }


def effective_parameters(model):
    """Describe dense HF weights, excluding adapter-only keys."""
    layers = dict(adapter_layers(model))
    for name, parameter in model.state_dict().items():
        if ".lora_" in name:
            continue
        layer = None
        if name.endswith(".base_layer.weight"):
            layer = layers[name.removesuffix(".base_layer.weight")]
        dense_name = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
        yield dense_name, parameter, layer


@torch.no_grad()
def effective_weights(model):
    """Yield W + BA without mutating W or unloading training adapters."""
    for name, parameter, layer in effective_parameters(model):
        value = full_tensor(parameter)
        if layer is not None:
            value = value.float() + adapter_delta(layer)
        yield name, value
