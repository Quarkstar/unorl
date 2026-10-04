"""Multi-update fixed-plane refresh with checkpointed transition state."""

import json
import math
from pathlib import Path

import ray
import torch
import torch.distributed as dist

from unorl.gradual_refresh import apply_plane_refresh
from unorl.low_resource import adapter_layers, full_tensor
from unorl.refresh_transition import (
    accumulate_plane_correction,
    make_refresh_plane,
    rotate_in_refresh_plane,
)
from unorl.relora import distribution_shift, response_log_distribution
from unorl.relora_refresh_worker import RefreshPolicyWorker


class GradualRefreshPolicyWorker(RefreshPolicyWorker):
    def _transition_offset(self, step):
        first = self.cfg.relora_first_merge_step
        first = self.merge_interval if first is None else first
        if not self.cfg.relora_enable_merge or step < first:
            return None
        offset = (step - first) % self.merge_interval
        return offset if offset < self.cfg.relora_refresh_updates else None

    def _should_merge(self, step):
        return self._transition_offset(step) is not None

    def _clear_transition(self):
        self._transition_start = None
        self._transition_count = 0
        self._transition_planes = {}
        self._transition_columns = {}
        self._transition_history_positions = {}

    def init_model(self, model_path, num_training_steps=None):
        result = super().init_model(model_path, num_training_steps)
        self._clear_transition()
        if dist.get_rank() == 0:
            path = Path(self.cfg.export_path).parent / "initial-adapter-audit.json"
            audit = json.loads(path.read_text())
            audit.update(
                {
                    "method": "multi-update fixed-plane refresh with warm B",
                    "refresh_updates": self.cfg.relora_refresh_updates,
                    "increment_plane_angle_degrees": self.cfg.relora_refresh_angle_degrees
                    / self.cfg.relora_refresh_updates,
                    "optimizer_reset": "none; preserve A moments and counters, project B first moment by actual row-change cosine, retain B variance",
                    "transition_checkpoint": "fixed planes, start step, increment count, compressed correction columns and history positions",
                }
            )
            path.write_text(json.dumps(audit, indent=2) + "\n")
        return result

    def _transition_state(self):
        return {
            "version": 1,
            "schedule": {
                "first": self.cfg.relora_first_merge_step,
                "interval": self.merge_interval,
                "updates": self.cfg.relora_refresh_updates,
                "total_angle": self.cfg.relora_refresh_angle_degrees,
                "enabled": self.cfg.relora_enable_merge,
            },
            "start": self._transition_start,
            "count": self._transition_count,
            "planes": self._transition_planes,
            "columns": self._transition_columns,
            "history_positions": self._transition_history_positions,
        }

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
                "gradual_refresh_transition": self._transition_state(),
            },
        )

    def _restore_transition(self, state, step):
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Missing or unsupported gradual-refresh checkpoint state")
        if state.get("schedule") != self._transition_state()["schedule"]:
            raise ValueError("Gradual-refresh checkpoint schedule differs from configuration")
        offset = self._transition_offset(step)
        active = offset is not None and offset + 1 < self.cfg.relora_refresh_updates
        expected_count = offset + 1 if active else 0
        if type(state.get("count")) is not int or state["count"] != expected_count:
            raise ValueError("Transition increment count disagrees with checkpoint step")
        fields = ("planes", "columns", "history_positions")
        if any(not isinstance(state.get(key), dict) for key in fields):
            raise ValueError("Invalid transition tensor mappings")
        if not active:
            if state.get("start") is not None or any(state[key] for key in fields):
                raise ValueError("Inactive transition must have no saved plane or history position")
            self._clear_transition()
            return
        if type(state.get("start")) is not int or state["start"] != step - offset:
            raise ValueError("Transition start disagrees with checkpoint step")
        layers = dict(adapter_layers(self.model.model))
        if set(state["planes"]) != set(layers):
            raise ValueError("Transition plane names differ from model adapter layers")
        rank_zero = dist.get_rank() == 0
        expected_history_names = set(layers) if rank_zero else set()
        if any(
            set(state[key]) != expected_history_names for key in ("columns", "history_positions")
        ):
            raise ValueError("Transition diagnostic mappings disagree with rank ownership")
        for name, layer in layers.items():
            plane = state["planes"][name]
            if len(plane) != 2 or any(
                row.device.type != "cpu" or row.dtype != torch.float32 for row in plane
            ):
                raise ValueError("Saved planes must contain two CPU float32 rows")
            expected_shape = tuple(layer.lora_A["default"].weight.shape)
            if any(tuple(row.shape) != expected_shape for row in plane):
                raise ValueError("Saved plane shape differs from adapter")
            rotate_in_refresh_plane(plane[0], plane, 0)
            if rank_zero:
                columns = state["columns"][name]
                shape = tuple(layer.lora_B["default"].weight.shape)
                if len(columns) != 2 or any(
                    tuple(column.shape) != shape
                    or column.device.type != "cpu"
                    or column.dtype != torch.float32
                    or not torch.isfinite(column).all()
                    for column in columns
                ):
                    raise ValueError("Invalid accumulated correction columns")
                index = state["history_positions"][name]
                history = self._factor_history.get(name, [])
                if type(index) is not int or index < 0 or index + 2 != len(history):
                    raise ValueError("Invalid transition position in correction history")
                for pair, row, column in zip(history[index:], plane, columns):
                    if not torch.equal(pair[0], row) or not torch.equal(pair[1], column):
                        raise ValueError("Compressed history differs from saved transition")
        self._transition_start = state["start"]
        self._transition_count = state["count"]
        self._transition_planes = state["planes"]
        self._transition_columns = state["columns"]
        self._transition_history_positions = state["history_positions"]

    def load_checkpoint(self, ckpt_dir, load_optimizer_states=True, load_lr_scheduler_states=True):
        states = super().load_checkpoint(ckpt_dir, load_optimizer_states, load_lr_scheduler_states)
        self._restore_transition(
            states.get("client_state", {}).get("gradual_refresh_transition"),
            self.scheduler.last_epoch,
        )
        # A live worker may have already synchronized different base weights.
        self._pending_base_sync = True
        self._math_probe = None
        self._before_step_factors = None
        self._step_update_metrics = None
        return states

    @torch.no_grad()
    def _begin_transition(self, step):
        if self._transition_start is not None:
            raise RuntimeError("Cannot start a new transition before finishing the previous one")
        self._transition_start = step
        for index, (name, layer) in enumerate(adapter_layers(self.model.model), start=1):
            a = full_tensor(layer.lora_A["default"].weight).cpu().float()
            plane = make_refresh_plane(a, seed=self.cfg.seed + step * 10000 + index)
            self._transition_planes[name] = plane
            # All ranks participate in the collective; diagnostics belong to rank zero.
            b = full_tensor(layer.lora_B["default"].weight)
            if dist.get_rank() != 0:
                continue
            b = b.cpu().float()
            columns = (torch.zeros_like(b), torch.zeros_like(b))
            self._transition_columns[name] = columns
            history = self._factor_history.setdefault(name, [])
            self._transition_history_positions[name] = len(history)
            history.extend(zip(plane, columns))

    @torch.no_grad()
    def _merge_with_probe(self, step):
        offset = self._transition_offset(step)
        if offset is None or self._math_probe is None:
            raise RuntimeError("A scheduled increment requires a captured real trajectory")
        if offset == 0:
            self._begin_transition(step)
        if self._transition_start != step - offset or self._transition_count != offset:
            raise RuntimeError("Transition state disagrees with the scheduled increment")
        update_metrics = (
            self._measure_delta() if getattr(self, "_before_step_factors", None) is not None else {}
        )
        tokens, length = self._math_probe
        before = response_log_distribution(self.model.model, tokens, length)

        def capture(name, a, fresh, b, plane, scale):
            if dist.get_rank() == 0:
                columns = accumulate_plane_correction(
                    self._transition_columns[name],
                    b.cpu(),
                    a.cpu(),
                    fresh.cpu(),
                    self._transition_planes[name],
                    scale,
                )
                self._transition_columns[name] = columns
                index = self._transition_history_positions[name]
                self._factor_history[name][index : index + 2] = list(
                    zip(self._transition_planes[name], columns)
                )

        metrics = apply_plane_refresh(
            self.model.model,
            self.optimizer,
            self._transition_planes,
            self.cfg.relora_refresh_angle_degrees / self.cfg.relora_refresh_updates,
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
        self._transition_count += 1
        metrics["refresh/transition_increment"] = float(self._transition_count)
        metrics["refresh/transition_start_step"] = float(self._transition_start)
        metrics.update(self._rank_diagnostics(step, commit=False))
        metrics.update(update_metrics)
        if dist.get_rank() == 0:
            root = Path(self.cfg.export_path).parent
            temporary = root / "relora-rank-factors.tmp"
            torch.save(self._factor_history, temporary)
            temporary.replace(root / "relora-rank-factors.pt")
        if self._transition_count == self.cfg.relora_refresh_updates:
            self._clear_transition()
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite multi-update refresh diagnostics")
        self._math_probe = None
        return metrics

    def resource_metrics(self):
        metrics = super().resource_metrics()
        first = self.cfg.relora_first_merge_step
        first = self.merge_interval if first is None else first
        relative = self.scheduler.last_epoch - first
        cycles = (
            0
            if relative < 0 or not self.cfg.relora_enable_merge
            else (
                relative // self.merge_interval
                + int(relative % self.merge_interval >= self.cfg.relora_refresh_updates - 1)
            )
        )
        metrics["relora/completed_cycles"] = float(cycles)
        metrics["refresh/completed_increments"] = float(
            cycles * self.cfg.relora_refresh_updates + self._transition_count
        )
        return metrics


PolicyWorker = ray.remote(num_gpus=1)(GradualRefreshPolicyWorker)
