"""Native AdamW ReLoRA with response-prefix probes and restart scheduling."""

import json
import math
from functools import partial
from pathlib import Path
from unittest.mock import patch

import ray
import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

from unorl.low_resource import adapter_layers, full_tensor, merge_and_reset
from unorl.merged_weights import MergedExportStrategy
from unorl.nora_merge_worker import NoRAMergePolicyWorker
from unorl.relora import (
    accumulated_spectrum,
    distribution_shift,
    response_log_distribution,
    restart_multiplier,
    trajectory_prefix,
)


class ReLoRAStrategy(MergedExportStrategy):
    def __init__(self, *args, merge_interval, restart_warmup_updates, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_interval = merge_interval
        self.restart_warmup_updates = restart_warmup_updates

    def _fsdp_init_train_model(self, model, optimizer, scheduler):
        wrapped, optimizer, _ = super()._fsdp_init_train_model(model, optimizer, scheduler)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda completed: restart_multiplier(
                completed, self.merge_interval, self.restart_warmup_updates
            ),
        )
        return wrapped, optimizer, scheduler


class ReLoRAPolicyWorker(NoRAMergePolicyWorker):
    @property
    def merge_interval(self):
        return self.cfg.relora_merge_interval

    @property
    def merge_label(self):
        return "ReLoRA"

    def init_model(self, model_path, num_training_steps=None):
        factory = partial(
            ReLoRAStrategy,
            merge_interval=self.merge_interval,
            restart_warmup_updates=self.cfg.relora_restart_warmup_updates,
        )
        with patch.object(fsdp_worker, "FSDPStrategy", factory):
            # Standard Kaiming wrapper; no NoRA/FA/LoFT optimizer substitution.
            result = fsdp_worker.FSDPPolicyWorkerBase.init_model(
                self, model_path, num_training_steps
            )
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise RuntimeError("ReLoRA comparison requires native AdamW")
        self._pending_base_sync = True
        self._memory_window_active = False
        self.merge_metrics = {}
        self._math_probe = None
        self._factor_history = {}
        if dist.get_rank() == 0:
            cache = Path(self.cfg.export_path).parent / "relora-rank-factors.pt"
            if cache.exists():
                self._factor_history = torch.load(cache, map_location="cpu", weights_only=True)
            audit = {
                "method": "ReLoRA restart-ramp comparison",
                "rank": 1,
                "alpha": 32,
                "initialization": "kaiming",
                "optimizer": type(self.optimizer).__name__,
                "betas": list(self.optimizer.param_groups[0]["betas"]),
                "epsilon": self.optimizer.param_groups[0]["eps"],
                "trainable_parameters": sum(
                    p.numel() for p in self.model.model.parameters() if p.requires_grad
                ),
                "merge_interval": self.merge_interval,
                "restart_warmup_updates": self.cfg.relora_restart_warmup_updates,
                "optimizer_reset": "clear moments and bias-correction counters; scheduler retained",
                "rollout_sync": "base W then adapter after merge/resume; adapter only otherwise",
                "probe": "one real math response prefix per rank, at most 128 response tokens",
                "rank_diagnostic": "thin QR / small SVD of accumulated low-rank factors",
            }
            (cache.parent / "initial-adapter-audit.json").write_text(
                json.dumps(audit, indent=2) + "\n"
            )
            print("ReLoRA initialized: " + json.dumps(audit), flush=True)
        return result

    def forward_backward(self, data, *args, **kwargs):
        if (self.scheduler.last_epoch + 1) % self.merge_interval == 0:
            self._math_probe = trajectory_prefix(data, self.cfg.relora_probe_response_tokens)
        return super().forward_backward(data, *args, **kwargs)

    @torch.no_grad()
    def _rank_diagnostics(self, step, commit):
        spectra = {}
        for name, layer in adapter_layers(self.model.model):
            a = full_tensor(layer.lora_A["default"].weight).float()
            b = full_tensor(layer.lora_B["default"].weight).float() * layer.scaling["default"]
            if dist.get_rank() == 0:
                pair = (a.cpu().clone(), b.cpu().clone())
                history = self._factor_history.setdefault(name, [])
                spectra[name] = accumulated_spectrum(history + [pair])
                if commit:
                    history.append(pair)
        metrics = {}
        if dist.get_rank() == 0:
            root = Path(self.cfg.export_path).parent
            with (root / "rank-diagnostics.jsonl").open("a") as handle:
                handle.write(
                    json.dumps({"step": step, "committed_merge": commit, "layers": spectra}) + "\n"
                )
            metrics = {
                "relora/mean_accumulated_stable_rank": sum(
                    s["stable_rank"] for s in spectra.values()
                )
                / len(spectra),
                "relora/mean_energy_outside_first_direction": sum(
                    s["energy_outside_first_direction"] for s in spectra.values()
                )
                / len(spectra),
            }
            if commit:
                temporary = root / "relora-rank-factors.tmp"
                torch.save(self._factor_history, temporary)
                temporary.replace(root / "relora-rank-factors.pt")
        return metrics

    @torch.no_grad()
    def _merge_with_probe(self, step):
        if self._math_probe is None:
            raise RuntimeError("A merge requires a captured math trajectory")
        tokens, response_length = self._math_probe
        before = response_log_distribution(self.model.model, tokens, response_length)
        metrics = self._rank_diagnostics(step, commit=True)
        state_entries = len(self.optimizer.state)
        metrics.update(
            merge_and_reset(
                self.model.model, seed=self.cfg.seed + step * 10000, init_method="kaiming"
            )
        )
        self.optimizer.state.clear()
        after = response_log_distribution(self.model.model, tokens, response_length)
        shift = distribution_shift(before, after, tokens[-response_length:])
        count_key = "relora/probe_response_tokens"
        max_key = "relora/probe_chosen_logprob_max_abs_diff"
        count = shift[count_key]
        for key, value in shift.items():
            tensor = torch.tensor(
                value if key in (count_key, max_key) else value * count,
                device=before.device,
                dtype=torch.float64,
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX if key == max_key else dist.ReduceOp.SUM)
            metrics[key] = tensor.item()
        for key in shift:
            if key not in (count_key, max_key):
                metrics[key] /= metrics[count_key]
        metrics.update(
            {
                "relora/retired_optimizer_state_entries": float(state_entries),
                "relora/optimizer_state_entries_after_reset": float(len(self.optimizer.state)),
                "relora/next_update_lr": self.scheduler.get_last_lr()[0],
            }
        )
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("Nonfinite ReLoRA boundary diagnostics")
        self._math_probe = None
        return metrics

    def optim_step(self):
        norm = super().optim_step()
        if (
            self.scheduler.last_epoch == self.cfg.max_training_steps
            and self.scheduler.last_epoch % self.merge_interval
        ):
            self.merge_metrics.update(
                self._rank_diagnostics(self.scheduler.last_epoch, commit=False)
            )
        return norm

    def resource_metrics(self):
        return {
            **super().resource_metrics(),
            "relora/next_update_lr": self.scheduler.get_last_lr()[0],
            "relora/restart_lr_multiplier": restart_multiplier(
                self.scheduler.last_epoch,
                self.merge_interval,
                self.cfg.relora_restart_warmup_updates,
            ),
        }


PolicyWorker = ray.remote(num_gpus=1)(ReLoRAPolicyWorker)
