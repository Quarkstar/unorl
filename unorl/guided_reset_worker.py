"""Three identical-prefix GRPO branches, with one training-batch calibration."""

import json
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from unorl.guided_reset import GradientSketch, choose_direction, fresh_probes, update_spaces
from unorl.low_resource import adapter_layers, copy_parameter, full_tensor, merge_and_reset
from unorl.prepared_hooks import PreparedProjectionHook
from unorl.relora import distribution_shift, response_log_distribution, trajectory_prefix
from unorl.relora_refresh_worker import RefreshPolicyWorker
from unorl.relora_worker import ReLoRAPolicyWorker


class GuidedResetPolicyWorker(RefreshPolicyWorker):
    """Calibrate before updates 41/81, then replay that batch for one update.

    Calibration changes no weights, Adam states, scheduler, or RNG. Every
    branch pays for the same extra forward/backward and sketch reduction.
    Standard retains factors/moments; random and guided merge then clear Adam.
    Both reset B to zero, with matched seeded Kaiming A norm. Only guided A's
    direction differs. Native clipping, AdamW and inference sync are retained.
    """

    def _should_merge(self, step):
        # Boundaries run before the NEXT update, not after optim_step.
        return False

    def init_model(self, model_path, num_training_steps=None):
        result = ReLoRAPolicyWorker.init_model(self, model_path, num_training_steps)
        self._previous_delta = self._before_step_factors = self._step_update_metrics = None
        self._boundary_metrics = {}
        self._calibrated_boundary = None
        if dist.get_rank() == 0:
            path = Path(self.cfg.export_path).parent / "initial-adapter-audit.json"
            audit = json.loads(path.read_text())
            audit.update(
                {
                    "method": "shared-prefix " + self.cfg.boundary_branch,
                    "calibration_steps": self.cfg.boundary_calibration_steps,
                    "calibration": "extra backward on the complete fresh training batch; gradients discarded before one native update on that same batch",
                    "optimizer_reset": "none"
                    if self.cfg.boundary_branch == "standard"
                    else "clear all Adam moments/counters; preserve constant scheduler",
                    "gradient_probe_width": self.cfg.boundary_probe_width,
                    "direction_selection": "top singular vector of training-loss gradient sketch outside accumulated input/output spaces; no held-out data",
                    "guarantee": "no guarantee of rank growth, reward gain, or optimality under Adam",
                }
            )
            path.write_text(json.dumps(audit, indent=2) + "\n")
        return result

    def load_checkpoint(self, *args, **kwargs):
        states = super().load_checkpoint(*args, **kwargs)
        if self.scheduler.last_epoch != 40 or any(self._factor_history.values()):
            raise ValueError("Expected the shared unmerged step-40 checkpoint")
        return states

    @torch.no_grad()
    def _prepare_observers(self, step):
        # The rank-0 factor history is the actual accumulated merged update.
        payload = [self._factor_history if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(payload, src=0, device=torch.cuda.current_device())
        history = payload[0]
        observers, hooks, spaces = {}, [], {}
        for index, (name, layer) in enumerate(adapter_layers(self.model.model)):
            a = full_tensor(layer.lora_A["default"].weight).float().clone()
            b = full_tensor(layer.lora_B["default"].weight).float().clone()
            factors = [(aa.to(a.device), bb.to(b.device)) for aa, bb in history.get(name, [])] + [
                (a, b * layer.scaling["default"])
            ]
            u, v, _ = update_spaces(factors)
            probes = fresh_probes(
                v,
                a.shape[1],
                self.cfg.boundary_probe_width,
                self.cfg.seed + step * 10000 + index,
                a.device,
            )
            observer = GradientSketch(probes, b.shape[0], self.cfg.boundary_token_chunk)
            observers[name], spaces[name] = observer, u
            hooks.append(PreparedProjectionHook(layer, observer))
        return observers, hooks, spaces

    def _calibrate(self, data, args, kwargs):
        step = self.scheduler.last_epoch
        observers, hooks, spaces = self._prepare_observers(step)
        rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        try:
            # Bypass automatic merge/probe capture, but use the actual native
            # loss, microbatch accumulation, FSDP and normalization.
            super().forward_backward(data, *args, **kwargs)
        finally:
            for hook in hooks:
                hook.close()
            self.optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1])
        rows, directions = [], {}
        for name, observer in observers.items():
            if not observer.observations:
                raise RuntimeError(f"Training-gradient observer missed layer {name}")
            dist.all_reduce(observer.gradient)
            observer.gradient.div_(dist.get_world_size())
        branch = self.cfg.boundary_branch
        tokens, length = trajectory_prefix(data, self.cfg.relora_probe_response_tokens)
        before = response_log_distribution(self.model.model, tokens, length)
        if branch != "standard":
            self._rank_diagnostics(step, commit=True)
            metrics = merge_and_reset(
                self.model.model, seed=self.cfg.seed + step * 10000, init_method="kaiming"
            )
            for name, layer in adapter_layers(self.model.model):
                reference = full_tensor(layer.lora_A["default"].weight).float().clone()
                direction, diagnostics = choose_direction(observers[name], spaces[name], reference)
                directions[name] = direction
                rows.append({"layer": name, **diagnostics})
            if branch == "guided":
                for name, layer in adapter_layers(self.model.model):
                    copy_parameter(layer.lora_A["default"].weight, directions[name])
            self.optimizer.state.clear()
            self._pending_base_sync = True
        else:
            metrics = self._rank_diagnostics(step, commit=False)
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
        self._boundary_metrics = {
            **metrics,
            "boundary/calibration_backward_passes": 1.0,
            "boundary/completed_step": float(step),
            "boundary/guided_layers": float(sum(not r["fallback_to_random"] for r in rows))
            if branch == "guided"
            else 0.0,
        }
        if dist.get_rank() == 0:
            row = {
                "step": step,
                "branch": branch,
                "layers": rows,
                "optimizer_state_entries_after_boundary": len(self.optimizer.state),
                "scheduler_step_after_boundary": self.scheduler.last_epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "boundary_metrics": metrics,
            }
            path = Path(self.cfg.export_path).parent / "boundary-reset-audit.jsonl"
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                "Shared-boundary calibration: "
                + json.dumps({k: v for k, v in row.items() if k != "layers"}),
                flush=True,
            )
        self._calibrated_boundary = step

    def forward_backward(self, data, *args, **kwargs):
        if not self._memory_window_active:
            torch.cuda.reset_peak_memory_stats()
            self._memory_window_active = True
        if self.scheduler.last_epoch in self.cfg.boundary_calibration_steps:
            if self._calibrated_boundary != self.scheduler.last_epoch:
                self._calibrate(data, args, kwargs)
        return super().forward_backward(data, *args, **kwargs)

    def optim_step(self):
        norm = super().optim_step()
        self.merge_metrics.update(self._boundary_metrics)
        self._boundary_metrics = {}
        return norm

    def resource_metrics(self):
        metrics = super().resource_metrics()
        step = self.scheduler.last_epoch
        metrics["relora/completed_cycles"] = float(
            sum(step > boundary for boundary in self.cfg.boundary_calibration_steps)
            if self.cfg.boundary_branch != "standard"
            else 0
        )
        return metrics


PolicyWorker = ray.remote(num_gpus=1)(GuidedResetPolicyWorker)
