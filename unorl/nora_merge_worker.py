"""Full NoRA-init with periodic merge, fresh adapters and explicit Adam reset."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import ray
import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

from unorl.low_resource import merge_and_reset
from unorl.merged_weights import BaseWeightExtractor, MergedExportStrategy
from unorl.nora_worker import NoRAInitPolicyWorker


def reset_nora_adapter(model, optimizer, seed):
    """Retain Parameter objects and native AdamW; retire old coordinate history."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("NoRA merge/reset requires native AdamW")
    metrics = merge_and_reset(model, seed=seed, init_method="nora_init")
    metrics["relora/retired_optimizer_state_entries"] = float(len(optimizer.state))
    optimizer.state.clear()
    metrics["relora/optimizer_state_entries_after_reset"] = float(len(optimizer.state))
    return metrics


class NoRAMergePolicyWorker(NoRAInitPolicyWorker):
    @property
    def merge_interval(self):
        return self.cfg.nora_merge_interval

    @property
    def merge_label(self):
        return "NoRA"

    def init_model(self, model_path, num_training_steps=None):
        with patch.object(fsdp_worker, "FSDPStrategy", MergedExportStrategy):
            result = super().init_model(model_path, num_training_steps)
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise RuntimeError("Expected native AdamW")
        self._pending_base_sync = True
        self._memory_window_active = False
        self.merge_metrics = {}
        if dist.get_rank() == 0:
            audit = {
                "rank": 1,
                "alpha": 1,
                "initialization": "nora_init",
                "trainable_parameters": sum(
                    p.numel() for p in self.model.model.parameters() if p.requires_grad
                ),
                "optimizer": type(self.optimizer).__name__,
                "merge_interval": self.cfg.nora_merge_interval,
                "optimizer_reset": "clear old adapter Adam state; preserve scheduler and LR",
                "rollout_sync": "base W at startup/resume and merges; native adapter BA every step",
            }
            root = Path(self.cfg.export_path).parent
            (root / "initial-adapter-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
            print("NoRA merge/reset initialized: " + json.dumps(audit), flush=True)
        return result

    async def init_weight_sync_state(self, inference_engine_client, inference_engine_cfg):
        self.weight_extractor = BaseWeightExtractor(self.model.model)
        # Build the sender with dense metadata from the beginning.
        await fsdp_worker.PolicyWorkerBase.init_weight_sync_state(
            self, inference_engine_client, inference_engine_cfg
        )

    async def broadcast_to_inference_engines(
        self, inference_engine_client, inference_engine_cfg, model_id=None
    ):
        if self._pending_base_sync:
            self._is_lora = False
            try:
                await super().broadcast_to_inference_engines(
                    inference_engine_client, inference_engine_cfg, model_id
                )
            finally:
                self._is_lora = True
        await super().broadcast_to_inference_engines(
            inference_engine_client, inference_engine_cfg, model_id
        )
        if self._pending_base_sync and dist.get_rank() == 0:
            row = {"step": self.scheduler.last_epoch, "base_then_adapter": True}
            path = Path(self.cfg.export_path).parent / "backbone-sync-audit.jsonl"
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                self.merge_label + " backbone and adapter synchronized: " + json.dumps(row),
                flush=True,
            )
        self._pending_base_sync = False

    def forward_backward(self, *args, **kwargs):
        if not self._memory_window_active:
            torch.cuda.reset_peak_memory_stats()
            self._memory_window_active = True
        return super().forward_backward(*args, **kwargs)

    @torch.no_grad()
    def _merge_with_probe(self, step):
        model = self.model.model
        tokens = torch.tensor([[1, 2, 3, 4]], device=torch.cuda.current_device())
        before = model(tokens).logits.float()
        metrics = reset_nora_adapter(model, self.optimizer, seed=self.cfg.seed + step * 10000)
        after = model(tokens).logits.float()
        difference = after - before
        metrics["relora/probe_logits_max_abs_diff"] = difference.abs().max().item()
        metrics["relora/probe_logits_rms_diff"] = difference.square().mean().sqrt().item()
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("Nonfinite merge/reset diagnostics")
        return metrics

    def optim_step(self):
        norm = super().optim_step()
        step = self.scheduler.last_epoch
        self.merge_metrics = {}
        if step > 0 and step % self.merge_interval == 0 and (norm is None or math.isfinite(norm)):
            self.merge_metrics = self._merge_with_probe(step)
            self._pending_base_sync = True
        root = Path(self.cfg.export_path).parent
        row = {
            "step": step,
            "rank": dist.get_rank(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "window": "policy forward/backward, optimizer, and merge/reset when scheduled",
            "merged": bool(self.merge_metrics),
            **self.merge_metrics,
        }
        with (root / f"training-memory-rank{dist.get_rank()}.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if self.merge_metrics and dist.get_rank() == 0:
            with (root / "merge-audit.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(self.merge_label + " merge/reset: " + json.dumps(row), flush=True)
        self._memory_window_active = False
        return norm

    def resource_metrics(self):
        return {
            **self.merge_metrics,
            "relora/completed_cycles": float(self.scheduler.last_epoch // self.merge_interval),
            "optimizer/state_entries": float(len(self.optimizer.state)),
        }


PolicyWorker = ray.remote(num_gpus=1)(NoRAMergePolicyWorker)
