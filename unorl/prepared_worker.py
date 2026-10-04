"""Native AdamW GRPO with checkpoint-managed prepared adapter history."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

from unorl.low_resource import adapter_layers, copy_parameter, full_tensor
from unorl.prepared_hooks import PreparedProjectionHook
from unorl.prepared_moments import PreparedBasisMoments
from unorl.prepared_selection import select_normal_descent
from unorl.relora import distribution_shift, response_log_distribution
from unorl.relora_refresh import factor_inner
from unorl.relora_refresh_worker import RefreshPolicyWorker
from unorl.relora_worker import ReLoRAPolicyWorker


class PreparedPolicyWorker(RefreshPolicyWorker):
    @property
    def merge_label(self):
        return "Prepared-history ReLoRA"

    def init_model(self, model_path, num_training_steps=None):
        result = ReLoRAPolicyWorker.init_model(self, model_path, num_training_steps)
        self._previous_delta = self._before_step_factors = self._step_update_metrics = None
        self._prepared = {}
        self._prepared_hooks = {}
        self._prepared_target = self._prepared_start = None
        if len(self.optimizer.param_groups) != 1:
            raise ValueError("Prepared-history recipe requires one native AdamW parameter group")
        if dist.get_rank() == 0:
            path = Path(self.cfg.export_path).parent / "initial-adapter-audit.json"
            audit = json.loads(path.read_text())
            audit.update(
                {
                    "method": "prepared-history normal-descent adapter switches",
                    "preparation_updates": self.cfg.prepared_window_updates,
                    "min_total_descent_fraction": self.cfg.prepared_min_total_fraction,
                    "candidate_angles": self.cfg.prepared_angles,
                    "optimizer_reset": "replace selected-factor moments with observed fixed-window projected history and its own counter; preserve native AdamW and global constant scheduler",
                    "compensation": "W += scale * (B_old @ A_old - B_new @ A_new)",
                    "rank_scope": "candidate bases refresh per window; useful learned rank is measured, not guaranteed",
                    "activation_saving": "identity autograd save_for_backward, compatible with checkpoint discard/recompute",
                }
            )
            path.write_text(json.dumps(audit, indent=2) + "\n")
        return result

    def _close_preparation(self):
        for hook in self._prepared_hooks.values():
            hook.close()
        self._prepared_hooks = {}
        self._prepared = {}
        self._prepared_target = self._prepared_start = None

    @torch.no_grad()
    def _begin_preparation(self, target):
        if self._prepared:
            raise RuntimeError("Cannot replace an active preparation window")
        group = self.optimizer.param_groups[0]
        for index, (name, layer) in enumerate(adapter_layers(self.model.model)):
            generator = torch.Generator().manual_seed(self.cfg.seed + target * 10000 + index)

            def basis(factor):
                first = full_tensor(factor).double().reshape(-1, 1)
                if not torch.isfinite(first).all() or first.norm() == 0:
                    raise ValueError("Preparation requires finite nonzero trained A and B")
                first /= first.norm()
                fresh = torch.randn(first.shape, generator=generator, dtype=torch.float64)
                fresh = fresh.to(first.device)
                fresh -= first * (first.T @ fresh)
                return torch.cat((first, fresh / fresh.norm()), dim=1).float()

            observer = PreparedBasisMoments(
                basis(layer.lora_A["default"].weight),
                basis(layer.lora_B["default"].weight),
                *group["betas"],
                chunk_tokens=self.cfg.prepared_token_chunk,
            )
            self._prepared[name] = observer
            self._prepared_hooks[name] = PreparedProjectionHook(layer, observer)
        if not self._prepared:
            raise ValueError("No rank-one adapter layers to prepare")
        self._prepared_target = target
        self._prepared_start = self.scheduler.last_epoch

    def forward_backward(self, data, *args, **kwargs):
        if not self._memory_window_active:
            torch.cuda.reset_peak_memory_stats()
            self._memory_window_active = True
        next_update = self.scheduler.last_epoch + 1
        first = self.cfg.relora_first_merge_step or self.merge_interval
        target = (
            first
            + max(0, math.ceil((next_update - first) / self.merge_interval)) * self.merge_interval
        )
        if (
            not self._prepared
            and target <= self.cfg.max_training_steps
            and next_update >= target - self.cfg.prepared_window_updates + 1
        ):
            self._begin_preparation(target)
        return super().forward_backward(data, *args, **kwargs)

    def _finish_preparation_update(self, norm):
        if not self._prepared:
            return
        if norm is not None and not math.isfinite(norm):
            for observer in self._prepared.values():
                observer.discard_update()
            return
        multiplier = min(1.0, self.strategy.max_norm / (norm + 1e-6)) if norm is not None else 1.0
        for observer in self._prepared.values():
            if not observer.pending_microbatches:
                raise RuntimeError("An active adapter observer missed training backward")
            # Native FSDP averages DP gradients. Square the reduced full-update
            # projection, never individual microbatch/rank contributions.
            for gradient in (observer.ga, observer.gb):
                dist.all_reduce(gradient)
                gradient.div_(dist.get_world_size())
            observer.finish_update(multiplier)

    @torch.no_grad()
    def _merge_with_probe(self, step):
        if self._prepared_target != step or self._math_probe is None:
            raise RuntimeError("Prepared merge needs its matching window and real response probe")
        if any(s.count != self.cfg.prepared_window_updates for s in self._prepared.values()):
            raise RuntimeError("Incomplete preparation history at the scheduled boundary")
        tokens, length = self._math_probe
        before = response_log_distribution(self.model.model, tokens, length)
        group = self.optimizer.param_groups[0]
        rows = []
        correction_energy = rounding_energy = 0.0
        transferred = 0
        for name, layer in adapter_layers(self.model.model):
            observer = self._prepared[name]
            a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
            old_a, old_b = full_tensor(a).float().clone(), full_tensor(b).float().clone()
            scale = layer.scaling["default"]
            selected = select_normal_descent(
                observer,
                old_a.norm().item(),
                old_b.norm().item(),
                scale,
                epsilon=group["eps"],
                angles=self.cfg.prepared_angles,
                min_total_fraction=self.cfg.prepared_min_total_fraction,
            )
            row = {
                "layer": name,
                **selected["diagnostics"],
                "reference_a_abs_cosine_at_switch": abs(
                    torch.dot(old_a.reshape(-1).double(), observer.qin[:, 0].double()).item()
                )
                / old_a.double().norm().item(),
                "reference_b_abs_cosine_at_switch": abs(
                    torch.dot(old_b.reshape(-1).double(), observer.qout[:, 0].double()).item()
                )
                / old_b.double().norm().item(),
                "history_observations": observer.count,
                "transferred": False,
            }
            if selected["diagnostics"]["has_positive_normal_descent"]:
                mapped = observer.map_moments(selected["a_coeff"], selected["b_coeff"], scale)
                weight = layer.get_base_layer().weight
                base = full_tensor(weight).float()
                correction = (old_b * scale) @ old_a
                correction.addmm_(mapped["b"], mapped["a"], beta=1, alpha=-scale)
                updated = (base + correction).to(weight.dtype)
                if not torch.isfinite(updated).all():
                    raise FloatingPointError("Nonfinite prepared base compensation")
                rounding_energy += ((updated.float() - base) - correction).square().sum().item()
                factors = [(old_a, old_b * scale), (mapped["a"], -mapped["b"] * scale)]
                correction_energy += factor_inner(factors, factors)
                copy_parameter(weight, updated)
                copy_parameter(a, mapped["a"])
                copy_parameter(b, mapped["b"])
                for param, mkey, vkey in ((a, "ma", "va"), (b, "mb", "vb")):
                    copy_parameter(self.optimizer.state[param]["exp_avg"], mapped[mkey])
                    copy_parameter(self.optimizer.state[param]["exp_avg_sq"], mapped[vkey])
                    self.optimizer.state[param]["step"].fill_(mapped["step"])
                if dist.get_rank() == 0:
                    self._factor_history.setdefault(name, []).extend(
                        (aa.cpu().clone(), bb.cpu().clone()) for aa, bb in factors
                    )
                row["transferred"] = True
                transferred += 1
            rows.append(row)
        after = response_log_distribution(self.model.model, tokens, length)
        shift = distribution_shift(before, after, tokens[-length:])
        count_key = "relora/probe_response_tokens"
        max_key = "relora/probe_chosen_logprob_max_abs_diff"
        count = shift[count_key]
        for key, value in shift.items():
            value = value if key in (count_key, max_key) else value * count
            tensor = torch.tensor(value, device=before.device, dtype=torch.float64)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX if key == max_key else dist.ReduceOp.SUM)
            shift[key] = tensor.item()
        for key in shift:
            if key not in (count_key, max_key):
                shift[key] /= shift[count_key]
        metrics = {
            **shift,
            **self._rank_diagnostics(step, commit=False),
            "relora/merged_layers": float(transferred),
            "relora/rounding_relative_l2": math.sqrt(
                rounding_energy / max(correction_energy, 1e-30)
            ),
            "prepared/history_observations": float(self.cfg.prepared_window_updates),
            "prepared/min_total_descent_retained": min(
                r["total_descent_retained_fraction"] for r in rows
            ),
            "prepared/mean_normal_descent_per_lr": sum(
                r["predicted_normal_descent_per_lr"] for r in rows
            )
            / len(rows),
            "prepared/min_reference_a_abs_cosine": min(
                r["reference_a_abs_cosine_at_switch"] for r in rows
            ),
            "prepared/min_reference_b_abs_cosine": min(
                r["reference_b_abs_cosine_at_switch"] for r in rows
            ),
        }
        if dist.get_rank() == 0:
            root = Path(self.cfg.export_path).parent
            with (root / "prepared-selection-audit.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, "layers": rows}) + "\n")
            temporary = root / "relora-rank-factors.tmp"
            torch.save(self._factor_history, temporary)
            temporary.replace(root / "relora-rank-factors.pt")
        self._close_preparation()
        self._math_probe = None
        return metrics

    def optim_step(self):
        if self.cfg.relora_track_updates:
            self._before_step_factors = self._capture_factors()
        # Bypass inherited automatic merging: preparation must consume the
        # observed clipping multiplier before its moments are installed.
        norm = fsdp_worker.FSDPPolicyWorkerBase.optim_step(self)
        self._finish_preparation_update(norm)
        if norm is not None and not math.isfinite(norm):
            raise FloatingPointError("Native update skipped; prepared projections discarded")
        self.merge_metrics = self._measure_delta() if self.cfg.relora_track_updates else {}
        self._before_step_factors = None
        step = self.scheduler.last_epoch
        merged = self._should_merge(step)
        if merged:
            self.merge_metrics.update(self._merge_with_probe(step))
            self._pending_base_sync = (
                self.merge_metrics["relora/merged_layers"] > 0 or self._pending_base_sync
            )
        elif step == self.cfg.max_training_steps:
            self.merge_metrics.update(self._rank_diagnostics(step, commit=False))
        row = {
            "step": step,
            "rank": dist.get_rank(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "window": "policy forward/backward, native AdamW, preparation and scheduled transfer",
            "merged": merged and self.merge_metrics["relora/merged_layers"] > 0,
            **self.merge_metrics,
        }
        root = Path(self.cfg.export_path).parent
        with (root / f"training-memory-rank{dist.get_rank()}.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if merged and dist.get_rank() == 0:
            with (root / "merge-audit.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        self._memory_window_active = False
        return norm

    def save_checkpoint(self, ckpt_dir, tokenizer=None):
        if any(s.pending_microbatches for s in self._prepared.values()):
            raise RuntimeError("Worker checkpoints require a completed optimizer update")
        self.strategy.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ckpt_dir=ckpt_dir,
            node_local_rank=self.get_node_local_rank(),
            tokenizer=tokenizer,
            client_state={
                "relora_factor_history": self._factor_history,
                "relora_previous_delta": self._previous_delta,
                "prepared": {name: state.state_dict() for name, state in self._prepared.items()},
                "prepared_target": self._prepared_target,
                "prepared_start": self._prepared_start,
                "native_adam_steps": self._native_adam_steps(),
                "prepared_protocol": self._prepared_protocol(),
            },
        )

    def _native_adam_steps(self):
        return {
            name: float(self.optimizer.state[param]["step"])
            for name, param in self.model.model.named_parameters()
            if "step" in self.optimizer.state.get(param, {})
        }

    def _prepared_protocol(self):
        return {
            "version": 1,
            "seed": self.cfg.seed,
            "first_merge": self.cfg.relora_first_merge_step or self.merge_interval,
            "merge_interval": self.merge_interval,
            "window_updates": self.cfg.prepared_window_updates,
            "min_total_fraction": self.cfg.prepared_min_total_fraction,
            "angles": self.cfg.prepared_angles,
            "betas": list(self.optimizer.param_groups[0]["betas"]),
            "epsilon": self.optimizer.param_groups[0]["eps"],
            "constant_lr": self.optimizer.param_groups[0]["lr"],
        }

    def load_checkpoint(self, ckpt_dir, load_optimizer_states=True, load_lr_scheduler_states=True):
        if not load_optimizer_states or not load_lr_scheduler_states:
            raise ValueError("Prepared resume requires optimizer and scheduler state")
        expected_protocol = self._prepared_protocol()
        self._close_preparation()
        states = fsdp_worker.FSDPPolicyWorkerBase.load_checkpoint(self, ckpt_dir)
        client = states.get("client_state", {})
        if client.get("prepared_protocol") != expected_protocol:
            raise ValueError("Prepared checkpoint protocol does not match the current recipe")
        if (
            "native_adam_steps" not in client
            or self._native_adam_steps() != client["native_adam_steps"]
        ):
            raise ValueError("Prepared native Adam counters were not restored exactly")
        self._factor_history = client["relora_factor_history"]
        self._previous_delta = client["relora_previous_delta"]
        self._prepared_target = client["prepared_target"]
        self._prepared_start = client["prepared_start"]
        layers = dict(adapter_layers(self.model.model))
        if client["prepared"]:
            if set(client["prepared"]) != set(layers) or not (
                self._prepared_start < self.scheduler.last_epoch < self._prepared_target
            ):
                raise ValueError("Invalid active preparation window in worker checkpoint")
        elif self._prepared_target is not None or self._prepared_start is not None:
            raise ValueError("Empty preparation checkpoint has active window metadata")
        for name, state in client["prepared"].items():
            observer = PreparedBasisMoments.from_state_dict(
                state, device=f"cuda:{torch.cuda.current_device()}"
            )
            if observer.pending_microbatches or observer.count != (
                self.scheduler.last_epoch - self._prepared_start
            ):
                raise ValueError("Prepared checkpoint observations do not match its window")
            layer = layers[name]
            if (
                observer.qin.shape[0] != layer.lora_A["default"].weight.shape[1]
                or observer.qout.shape[0] != layer.lora_B["default"].weight.shape[0]
                or (observer.beta1, observer.beta2)
                != tuple(self.optimizer.param_groups[0]["betas"])
            ):
                raise ValueError(
                    "Prepared checkpoint bases/moments do not match the model/optimizer"
                )
            self._prepared[name] = observer
            self._prepared_hooks[name] = PreparedProjectionHook(layers[name], observer)
        self._pending_base_sync = True
        return states


PolicyWorker = ray.remote(num_gpus=1)(PreparedPolicyWorker)
