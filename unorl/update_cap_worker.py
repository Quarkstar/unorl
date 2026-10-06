"""Ordinary ReLoRA with a cycle-local bound on effective weight-update size."""

import json
import math
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from unorl.guided_reset_worker import GuidedResetPolicyWorker
from unorl.low_resource import adapter_layers, copy_parameter, full_tensor
from unorl.relora_refresh import factor_inner
from unorl.update_cap import cap_multiplier, interpolate_factors, update_norm


class UpdateCapPolicyWorker(GuidedResetPolicyWorker):
    def init_model(self, model_path, num_training_steps=None):
        result = super().init_model(model_path, num_training_steps)
        self._update_budget = None
        if dist.get_rank() == 0:
            path = Path(self.cfg.export_path).parent / "initial-adapter-audit.json"
            audit = json.loads(path.read_text())
            audit.update(
                {
                    "method": "ordinary ReLoRA with effective-weight update cap",
                    "update_budget": "norm of the last applied pre-reset update; frozen through the next cycle",
                    "update_budget_ratio": self.cfg.update_cap_budget_ratio,
                    "cap": "common scalar interpolation of native Adam-proposed A/B steps, accounting for their quadratic product",
                    "adam_state": "native gradient moments/counters unchanged by interpolation; reset normally at boundaries",
                    "schedule": "constant native scheduler; applied step multiplier recorded separately",
                }
            )
            path.write_text(json.dumps(audit, indent=2) + "\n")
        return result

    @torch.no_grad()
    def _capture_factors(self):
        # Small CPU factor snapshots on every rank allow identical local
        # interpolation. No dense W or optimizer moments are copied.
        return {
            name: (
                full_tensor(layer.lora_A["default"].weight).cpu().double().clone(),
                full_tensor(layer.lora_B["default"].weight).cpu().double().clone(),
                layer.scaling["default"],
            )
            for name, layer in adapter_layers(self.model.model)
        }

    def _calibrate(self, data, args, kwargs):
        budget = 0.0
        if dist.get_rank() == 0:
            if self._previous_delta is None:
                raise ValueError("The boundary checkpoint must contain its previous applied update")
            energy = sum(factor_inner(f, f) for f in self._previous_delta.values())
            budget = math.sqrt(max(0.0, energy)) * self.cfg.update_cap_budget_ratio
        value = torch.tensor(budget, device=torch.cuda.current_device(), dtype=torch.float64)
        dist.broadcast(value, src=0)
        self._update_budget = value.item()
        if not math.isfinite(self._update_budget) or self._update_budget <= 0:
            raise ValueError("The preceding applied weight update must be finite and nonzero")
        super()._calibrate(data, args, kwargs)
        self._boundary_metrics["update_cap/budget_l2"] = self._update_budget

    @torch.no_grad()
    def _measure_delta(self):
        if self._update_budget is None:
            raise RuntimeError("No cycle-local update budget was initialized")
        after = self._capture_factors()
        before = self._before_step_factors
        multiplier, diagnostics = cap_multiplier(before, after, self._update_budget)
        # Use rank zero's scalar even if reduction arithmetic differs slightly.
        value = torch.tensor(multiplier, device=torch.cuda.current_device(), dtype=torch.float64)
        dist.broadcast(value, src=0)
        multiplier = value.item()
        if multiplier < 1:
            applied = interpolate_factors(before, after, multiplier)
            for name, layer in adapter_layers(self.model.model):
                a, b, _ = applied[name]
                a_param = layer.lora_A["default"].weight
                b_param = layer.lora_B["default"].weight
                copy_parameter(a_param, a.to(device=a_param.device, dtype=a_param.dtype))
                copy_parameter(b_param, b.to(device=b_param.device, dtype=b_param.dtype))
        applied_norm = update_norm(before, self._capture_factors())
        if applied_norm > self._update_budget * (1 + 1e-5) + 1e-8:
            raise FloatingPointError("Applied update exceeds its cycle budget")
        metrics = super()._measure_delta()
        if self.scheduler.last_epoch == self.cfg.max_training_steps:
            metrics.update(super()._rank_diagnostics(self.scheduler.last_epoch, commit=False))
        metrics.update({f"update_cap/{key}": float(val) for key, val in diagnostics.items()})
        metrics.update(
            {"update_cap/step_multiplier": multiplier, "update_cap/applied_l2": applied_norm}
        )
        self._step_update_metrics = metrics
        if dist.get_rank() == 0:
            with (Path(self.cfg.export_path).parent / "update-cap-audit.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": self.scheduler.last_epoch, **metrics}) + "\n")
        return metrics

    def _rank_diagnostics(self, step, commit):
        # The parent's end-step diagnostic precedes cap interpolation. Emit
        # the final spectrum once, after the applied update is measured.
        if (
            not commit
            and step == self.cfg.max_training_steps
            and self._before_step_factors is not None
        ):
            return {}
        return super()._rank_diagnostics(step, commit)

    def optim_step(self):
        norm = super().optim_step()
        # Parent allocator logging precedes interpolation; include its gathers
        # and parameter writes in an additional exact allocator-peak record.
        row = {
            "step": self.scheduler.last_epoch,
            "rank": dist.get_rank(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "window": "calibration, native update, merge/reset, and update-cap interpolation",
        }
        path = Path(self.cfg.export_path).parent / f"update-cap-memory-rank{dist.get_rank()}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return norm


PolicyWorker = ray.remote(num_gpus=1)(UpdateCapPolicyWorker)
