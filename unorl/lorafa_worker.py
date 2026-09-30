"""LoRA-FA on SkyRL's native FSDP2/AdamW path, with training memory records."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import ray
import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

from unorl.lorafa import correct_gradients, correction_factors, freeze_lora_a


class LoRAFAWrapper(HFModelWrapper):
    def __init__(self, *args, **kwargs):
        if kwargs.get("lora_init_method") != "lorafa":
            raise ValueError("LoRA-FA worker requires lora.init_method=lorafa")
        # Method selector only: retain the reference Kaiming initialization.
        kwargs["lora_init_method"] = "kaiming"
        super().__init__(*args, **kwargs)
        freeze_lora_a(self.model)


class LoRAFAStrategy(FSDPStrategy):
    def _fsdp_init_train_model(self, model, optimizer, scheduler):
        result = super()._fsdp_init_train_model(model, optimizer, scheduler)
        self.lorafa_factors = correction_factors(result[0].model)
        return result

    def load_checkpoint(self, model, *args, **kwargs):
        result = super().load_checkpoint(model, *args, **kwargs)
        self.lorafa_factors = correction_factors(model.model)
        return result

    def optimizer_step(self, optimizer, model, scheduler, **kwargs):
        corrected = correct_gradients(model.model, self.lorafa_factors)
        if corrected != len(self.lorafa_factors):
            raise RuntimeError("LoRA-FA requires a gradient for every B projection")
        # Native clipping, nonfinite handling, AdamW, scheduler and zero_grad.
        return super().optimizer_step(optimizer, model, scheduler, **kwargs)


class LoRAFAPolicyWorker(fsdp_worker.FSDPPolicyWorkerBase):
    def init_model(self, model_path, num_training_steps=None):
        with (
            patch.object(fsdp_worker, "HFModelWrapper", LoRAFAWrapper),
            patch.object(fsdp_worker, "FSDPStrategy", LoRAFAStrategy),
        ):
            result = super().init_model(model_path, num_training_steps)
        model = self.model.model
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        factors = self.strategy.lorafa_factors
        audit = {
            "rank": self.cfg.policy.model.lora.rank,
            "alpha": self.cfg.policy.model.lora.alpha,
            "initialization": "kaiming",
            "A_frozen": True,
            "B_matrices": len(trainable),
            "trainable_parameters": sum(p.numel() for _, p in trainable),
            "optimizer": type(self.optimizer).__name__,
            "adam_betas": list(self.optimizer.param_groups[0]["betas"]),
            "adam_epsilon": self.optimizer.param_groups[0]["eps"],
            "correction_min": min(factors.values()),
            "correction_max": max(factors.values()),
            "gram_regularization": 1e-8,
            "correction_order": "after accumulation, before native clipping and AdamW",
        }
        print("LoRA-FA initialized: " + json.dumps(audit), flush=True)
        if dist.get_rank() == 0:
            root = Path(self.cfg.export_path).parent
            (root / "initial-adapter-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        self._memory_window_active = False
        return result

    def forward_backward(self, *args, **kwargs):
        if not self._memory_window_active:
            torch.cuda.reset_peak_memory_stats()
            self._memory_window_active = True
        return super().forward_backward(*args, **kwargs)

    def optim_step(self):
        norm = super().optim_step()
        row = {
            "step": self.scheduler.last_epoch,
            "rank": dist.get_rank(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "grad_norm_corrected_before_clip": norm,
            "window": "policy forward/backward through optimizer and zero_grad",
        }
        path = Path(self.cfg.export_path).parent / f"training-memory-rank{dist.get_rank()}.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        self._memory_window_active = False
        return norm


PolicyWorker = ray.remote(num_gpus=1)(LoRAFAPolicyWorker)
