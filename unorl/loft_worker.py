"""SkyRL FSDP2 worker using rank-one LoFT-simple and native adapter rollout."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import ray
import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from transformers import get_scheduler

from unorl.loft import LoFTSimpleAdamW


class LoFTWrapper(HFModelWrapper):
    def __init__(self, *args, **kwargs):
        if kwargs.get("lora_init_method") != "loft_simple":
            raise ValueError("LoFT worker requires init_method=loft_simple")
        kwargs["lora_init_method"] = "kaiming"
        super().__init__(*args, **kwargs)


class LoFTStrategy(FSDPStrategy):
    def _fsdp_init_train_model(self, model, optimizer, scheduler):
        wrapped, _, _ = super()._fsdp_init_train_model(model, optimizer, scheduler)
        cfg = self.optimizer_config
        module = wrapped.model if isinstance(wrapped, HFModelWrapper) else wrapped
        optimizer = LoFTSimpleAdamW(
            module, lr=cfg.lr, betas=cfg.adam_betas, eps=1e-4, weight_decay=cfg.weight_decay
        )
        scheduler = get_scheduler(
            cfg.scheduler,
            optimizer,
            num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=self.total_training_steps,
        )
        return wrapped, optimizer, scheduler

    def optimizer_step(self, optimizer, model, scheduler, **kwargs):
        norm = optimizer.calibrated_grad_norm()
        if not math.isfinite(norm):
            optimizer.zero_grad()
            return torch.tensor(norm)
        coef = min(1.0, self.max_norm / (norm + 1e-6)) if self.max_norm > 0 else 1.0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(coef)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return torch.tensor(norm)


class LoFTPolicyWorker(fsdp_worker.FSDPPolicyWorkerBase):
    def init_model(self, model_path, num_training_steps=None):
        with (
            patch.object(fsdp_worker, "HFModelWrapper", LoFTWrapper),
            patch.object(fsdp_worker, "FSDPStrategy", LoFTStrategy),
        ):
            result = super().init_model(model_path, num_training_steps)
        self._memory_window_active = False
        audit = {
            "method": "LoFT-simple",
            "rank": 1,
            "alpha": 1,
            "initialization": "kaiming",
            "trainable_parameters": sum(
                p.numel() for p in self.model.model.parameters() if p.requires_grad
            ),
            "optimizer": type(self.optimizer).__name__,
            "epsilon": 1e-4,
            "betas": list(self.optimizer.param_groups[0]["betas"]),
            "regularization_A": 1e-6,
            "regularization_B": 1e-8,
            "alternation": "B first; update moments for both factors every step",
            "first_moment_transport": True,
            "second_moment_transport": False,
            "clipping": "active calibrated factor gradients; scale both raw gradients",
            "merge_reset": False,
            "reference_commit": "148998d98f901cd7744db7baa5b2b4868fa16bf0",
        }
        if dist.get_rank() == 0:
            (Path(self.cfg.export_path).parent / "initial-adapter-audit.json").write_text(
                json.dumps(audit, indent=2) + "\n"
            )
            print("LoFT-simple initialized: " + json.dumps(audit), flush=True)
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
            "calibrated_active_grad_norm": norm,
            "window": "policy forward/backward through LoFT optimizer and zero_grad",
            **self.optimizer.last_metrics,
        }
        path = Path(self.cfg.export_path).parent / f"training-memory-rank{dist.get_rank()}.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        self._memory_window_active = False
        return norm

    def resource_metrics(self):
        return {
            **self.optimizer.last_metrics,
            "optimizer/state_entries": float(len(self.optimizer.state)),
            "loft/next_update_A": float(self.optimizer.update_A),
        }


PolicyWorker = ray.remote(num_gpus=1)(LoFTPolicyWorker)
