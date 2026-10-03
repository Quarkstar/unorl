"""ReLoRA with compensated gradual A refresh and warm native AdamW state."""

import json
import math
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from unorl.low_resource import adapter_layers, full_tensor
from unorl.relora import distribution_shift, response_log_distribution
from unorl.relora_refresh import factor_inner, refresh_adapter, update_factors
from unorl.relora_worker import ReLoRAPolicyWorker


class RefreshPolicyWorker(ReLoRAPolicyWorker):
    def init_model(self, model_path, num_training_steps=None):
        result = super().init_model(model_path, num_training_steps)
        self._previous_delta = None
        self._before_step_factors = None
        self._step_update_metrics = None
        if dist.get_rank() == 0:
            path = Path(self.cfg.export_path).parent / "initial-adapter-audit.json"
            audit = json.loads(path.read_text())
            audit.update(
                {
                    "method": "gradual A refresh with warm B"
                    if self.cfg.relora_enable_merge
                    else "standard LoRA continuation, no merges",
                    "refresh_angle_degrees": self.cfg.relora_refresh_angle_degrees,
                    "first_merge_step": self.cfg.relora_first_merge_step,
                    "optimizer_reset": "none; preserve A moments and all counters, project B first moment by cosine, retain B variance",
                    "compensation": "W += scale * B @ (A_old - A_new); B unchanged",
                    "moment_transport": "approximate for B; unknown orthogonal-direction variance is not reconstructed",
                }
            )
            path.write_text(json.dumps(audit, indent=2) + "\n")
            print("Gradual refresh initialized: " + json.dumps(audit), flush=True)
        return result

    def save_checkpoint(self, ckpt_dir, tokenizer=None):
        self.strategy.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ckpt_dir=ckpt_dir,
            node_local_rank=self.get_node_local_rank(),
            tokenizer=tokenizer,
            client_state={
                "relora_factor_history": self._factor_history,
                "relora_previous_delta": getattr(self, "_previous_delta", None),
            },
        )

    def load_checkpoint(self, ckpt_dir, load_optimizer_states=True, load_lr_scheduler_states=True):
        if not load_optimizer_states or not load_lr_scheduler_states:
            raise ValueError("The matched continuation requires optimizer and scheduler history")
        states = super().load_checkpoint(ckpt_dir, load_optimizer_states, load_lr_scheduler_states)
        self._factor_history = states.get("client_state", {}).get("relora_factor_history", {})
        self._previous_delta = states.get("client_state", {}).get("relora_previous_delta")
        counters = [
            float(state["step"]) for state in self.optimizer.state.values() if "step" in state
        ]
        step = self.scheduler.last_epoch
        if not counters or min(counters) != step or max(counters) != step:
            raise ValueError("Loaded Adam counters must match the checkpoint scheduler step")
        if dist.get_rank() == 0:
            audit = {
                "checkpoint": str(ckpt_dir),
                "scheduler_step": step,
                "adam_step_min": min(counters),
                "adam_step_max": max(counters),
                "optimizer": type(self.optimizer).__name__,
                "lr": self.optimizer.param_groups[0]["lr"],
                "history_layers": len(self._factor_history),
                "history_restored_from_checkpoint": True,
            }
            (Path(self.cfg.export_path).parent / "resume-policy-audit.json").write_text(
                json.dumps(audit, indent=2) + "\n"
            )
            print("Refresh continuation resume audit: " + json.dumps(audit), flush=True)
        return states

    @torch.no_grad()
    def _capture_factors(self):
        factors = {}
        for name, layer in adapter_layers(self.model.model):
            a = full_tensor(layer.lora_A["default"].weight)
            b = full_tensor(layer.lora_B["default"].weight)
            if dist.get_rank() == 0:
                factors[name] = (a.cpu().double(), b.cpu().double(), layer.scaling["default"])
        return factors

    def _measure_delta(self):
        after = self._capture_factors()
        metrics = {}
        if dist.get_rank() == 0:
            current = {}
            for name, (a, b, scale) in after.items():
                old_a, old_b, _ = self._before_step_factors[name]
                current[name] = update_factors(old_a, old_b, a, b, scale)
            energy = sum(factor_inner(factors, factors) for factors in current.values())
            if not math.isfinite(energy):
                raise FloatingPointError("Nonfinite effective weight-space update")
            metrics["updates/effective_delta_l2"] = math.sqrt(max(0.0, energy))
            if self._previous_delta is not None:
                old_energy = sum(
                    factor_inner(factors, factors) for factors in self._previous_delta.values()
                )
                dot = sum(
                    factor_inner(factors, self._previous_delta[name])
                    for name, factors in current.items()
                )
                metrics["updates/cosine_with_previous_delta"] = max(
                    -1.0, min(1.0, dot / math.sqrt(max(energy * old_energy, 1e-30)))
                )
            self._previous_delta = current
        self._step_update_metrics = metrics
        return metrics

    def optim_step(self):
        self._step_update_metrics = None
        if self.cfg.relora_track_updates:
            self._before_step_factors = self._capture_factors()
        norm = super().optim_step()
        if self.cfg.relora_track_updates:
            if self._step_update_metrics is None:
                self._measure_delta()
            self.merge_metrics.update(self._step_update_metrics)
            self._before_step_factors = None
        return norm

    @torch.no_grad()
    def _merge_with_probe(self, step):
        if self._math_probe is None:
            raise RuntimeError("Gradual refresh requires a captured real trajectory")
        update_metrics = (
            self._measure_delta() if getattr(self, "_before_step_factors", None) is not None else {}
        )
        tokens, length = self._math_probe
        before = response_log_distribution(self.model.model, tokens, length)

        def capture(name, a, b):
            if dist.get_rank() == 0:
                self._factor_history.setdefault(name, []).append((a.cpu().clone(), b.cpu().clone()))

        metrics = refresh_adapter(
            self.model.model,
            self.optimizer,
            self.cfg.seed + step * 10000,
            self.cfg.relora_refresh_angle_degrees,
            on_correction=capture,
        )
        after = response_log_distribution(self.model.model, tokens, length)
        shift = distribution_shift(before, after, tokens[-length:])
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
        metrics.update(self._rank_diagnostics(step, commit=False))
        metrics.update(update_metrics)
        if dist.get_rank() == 0:
            root = Path(self.cfg.export_path).parent
            temporary = root / "relora-rank-factors.tmp"
            torch.save(self._factor_history, temporary)
            temporary.replace(root / "relora-rank-factors.pt")
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite gradual-refresh diagnostics")
        self._math_probe = None
        return metrics

    def resource_metrics(self):
        metrics = super().resource_metrics()
        first = self.cfg.relora_first_merge_step or self.merge_interval
        step = self.scheduler.last_epoch
        metrics["relora/completed_cycles"] = float(
            max(0, (step - first) // self.merge_interval + 1) if self.cfg.relora_enable_merge else 0
        )
        return metrics


PolicyWorker = ray.remote(num_gpus=1)(RefreshPolicyWorker)
