"""Native eight-rank prepared-gradient collection, moment replacement and resume."""

import argparse
import faulthandler
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.train.config import SkyRLTrainConfig
from transformers import Qwen3Config, Qwen3ForCausalLM

from unorl.low_resource import adapter_layers, copy_parameter, effective_weights, full_tensor
from unorl.prepared_hooks import PreparedProjectionHook
from unorl.prepared_moments import PreparedBasisMoments
from unorl.relora_worker import ReLoRAStrategy


def relative_error(actual, expected):
    return (actual.double() - expected.double()).norm().item() / max(
        expected.double().norm().item(), 1e-30
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(90, repeat=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    hooks = {}
    try:
        model_path = args.output / "tiny-model"
        if dist.get_rank() == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(42)
            Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=8,
                    tie_word_embeddings=False,
                )
            ).save_pretrained(model_path)
        dist.barrier()
        cfg = SkyRLTrainConfig.from_cli_overrides(
            [
                "trainer.policy.model.lora.rank=1",
                "trainer.policy.model.lora.alpha=32",
                "trainer.policy.optimizer_config.lr=0.001",
                "trainer.policy.optimizer_config.num_warmup_steps=0",
                'trainer.policy.optimizer_config.scheduler="constant"',
                "trainer.policy.optimizer_config.weight_decay=0.0",
                "trainer.policy.optimizer_config.max_grad_norm=0.001",
            ]
        )
        strategy = ReLoRAStrategy(
            merge_interval=100,
            restart_warmup_updates=0,
            fsdp_config=cfg.trainer.policy.fsdp_config,
            optimizer_config=cfg.trainer.policy.optimizer_config,
            model_config=cfg.trainer.policy.model,
            num_training_steps=6,
        )
        strategy.setup_distributed()
        wrapped = HFModelWrapper(
            str(model_path),
            use_flash_attention_2=False,
            bf16=False,
            lora_rank=1,
            lora_alpha=32,
            lora_init_method="kaiming",
            target_modules="all-linear",
            remove_microbatch_padding=False,
            use_torch_compile=False,
            meta_init=dist.get_rank() != 0,
        )
        wrapped.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        wrapped, optimizer, scheduler = strategy.prepare((wrapped, None, None))
        model = wrapped.model
        layers = dict(adapter_layers(model))
        identities = {n: id(p) for n, p in model.named_parameters()}
        assert isinstance(optimizer, torch.optim.AdamW)
        # Nonzero B exposes both current A and B gradients in this fixture.
        with torch.no_grad():
            for index, layer in enumerate(layers.values()):
                b = layer.lora_B["default"].weight
                generator = torch.Generator().manual_seed(150 + index)
                fresh = torch.randn(tuple(b.shape), generator=generator) * 0.005
                copy_parameter(b, fresh.cuda())
        observers, histories, oracle_current = {}, {}, {}
        oracle_handles = []
        gradient_projection_errors = []
        warm_moment_errors = []
        applied_lrs, clipping = [], []
        forward_differences = []

        def attach():
            for index, (name, layer) in enumerate(layers.items()):
                a = full_tensor(layer.lora_A["default"].weight).float()
                b = full_tensor(layer.lora_B["default"].weight).float()
                generator = torch.Generator().manual_seed(170 + index)

                def basis(direction):
                    first = direction.reshape(-1, 1) / direction.norm()
                    second = torch.randn(first.shape, generator=generator).cuda()
                    second -= first * (first.T @ second)
                    return torch.cat((first, second / second.norm()), dim=1)

                qin, qout = basis(a), basis(b)
                if dist.get_rank() == 0 and index == 0:
                    print(
                        json.dumps(
                            {
                                "basis_diagnostic": name,
                                "qin_gram": (qin.T @ qin).tolist(),
                                "qin_gram_double": (qin.double().T @ qin.double()).tolist(),
                                "qout_gram": (qout.T @ qout).tolist(),
                                "qout_gram_double": (qout.double().T @ qout.double()).tolist(),
                                "autocast_enabled": torch.is_autocast_enabled("cuda"),
                                "matmul_precision": torch.get_float32_matmul_precision(),
                            }
                        ),
                        flush=True,
                    )
                observers[name] = PreparedBasisMoments(qin, qout, chunk_tokens=2)
                hooks[name] = PreparedProjectionHook(layer, observers[name])
                histories[name] = []

                def forward_oracle(module, inputs, output, name=name):
                    if not torch.is_grad_enabled() or not output.requires_grad:
                        return
                    x = inputs[0].detach()

                    def capture(dy):
                        try:
                            # A separate oracle may construct dense G only in
                            # this tiny fixture, never in the actual collector.
                            gradient = dy.detach().float().reshape(
                                -1, dy.shape[-1]
                            ).T @ x.float().reshape(-1, x.shape[-1])
                            oracle_current[name].add_(gradient)
                        finally:
                            handle.remove()

                    handle = output.register_hook(capture)

                oracle_handles.append(layer.register_forward_hook(forward_oracle))

        def perform(step, refresh=False):
            applied_lrs.append(optimizer.param_groups[0]["lr"])
            if observers:
                for name, layer in layers.items():
                    oracle_current[name] = torch.zeros(
                        tuple(layer.get_base_layer().weight.shape), device="cuda"
                    )
            # Different rank-local data and unequal microbatch token counts.
            for length in (3, 5):
                ids = (
                    (torch.arange(length, device="cuda") + dist.get_rank() + step) % 30 + 1
                ).reshape(1, -1)
                loss = model(ids, labels=ids).loss * (length - 1) / 6
                strategy.backward(loss, wrapped, optimizer)
            if observers:
                for name, state in observers.items():
                    assert state.pending_microbatches == 2
                    for value in (state.ga, state.gb, oracle_current[name]):
                        dist.all_reduce(value)
                        value.div_(dist.get_world_size())
                    if step == 2:
                        layer = layers[name]
                        a = full_tensor(layer.lora_A["default"].weight).float()
                        b = full_tensor(layer.lora_B["default"].weight).float()
                        ga = layer.scaling["default"] * b.norm() * state.ga[:1]
                        gb = layer.scaling["default"] * a.norm() * state.gb[:, :1]
                        gradient_projection_errors.extend(
                            [
                                relative_error(
                                    ga, full_tensor(layer.lora_A["default"].weight.grad)
                                ),
                                relative_error(
                                    gb, full_tensor(layer.lora_B["default"].weight.grad)
                                ),
                            ]
                        )
            norm = float(strategy.optimizer_step(optimizer, wrapped, scheduler))
            assert torch.isfinite(torch.tensor(norm))
            multiplier = min(1.0, strategy.max_norm / (norm + 1e-6))
            clipping.append(multiplier)
            for name, state in observers.items():
                state.finish_update(multiplier)
                histories[name].append(oracle_current[name].clone() * multiplier)
            assert scheduler.last_epoch == step
            if refresh:
                before_weights = {n: v.clone() for n, v in effective_weights(model)}
                probe = torch.tensor([[1, 2, 3, 4]], device="cuda")
                with torch.no_grad():
                    before = model(probe).logits.float()
                    for name, layer in layers.items():
                        a = layer.lora_A["default"].weight
                        b = layer.lora_B["default"].weight
                        old_a = full_tensor(a).float().clone()
                        old_b = full_tensor(b).float().clone()
                        scale = layer.scaling["default"]
                        state = observers[name]
                        ac, bc = state.select_coefficients(old_a.norm().item(), old_b.norm().item())
                        mapped = state.map_moments(ac, bc, scale)
                        reference = [
                            torch.nn.Parameter(torch.zeros_like(mapped[key])) for key in ("a", "b")
                        ]
                        reference_optimizer = torch.optim.AdamW(reference, weight_decay=0)
                        for gradient in histories[name]:
                            reference[0].grad = scale * mapped["b"].T @ gradient
                            reference[1].grad = scale * gradient @ mapped["a"].T
                            reference_optimizer.step()
                        for param, mkey, vkey in zip(reference, ("ma", "mb"), ("va", "vb")):
                            warm_moment_errors.extend(
                                [
                                    relative_error(
                                        mapped[mkey], reference_optimizer.state[param]["exp_avg"]
                                    ),
                                    relative_error(
                                        mapped[vkey], reference_optimizer.state[param]["exp_avg_sq"]
                                    ),
                                ]
                            )
                        weight = layer.get_base_layer().weight
                        correction = (old_b * scale) @ old_a
                        correction.addmm_(mapped["b"], mapped["a"], beta=1, alpha=-scale)
                        copy_parameter(weight, full_tensor(weight).float() + correction)
                        copy_parameter(a, mapped["a"])
                        copy_parameter(b, mapped["b"])
                        for param, mkey, vkey in zip((a, b), ("ma", "mb"), ("va", "vb")):
                            copy_parameter(optimizer.state[param]["exp_avg"], mapped[mkey])
                            copy_parameter(optimizer.state[param]["exp_avg_sq"], mapped[vkey])
                            optimizer.state[param]["step"].fill_(mapped["step"])
                    after = model(probe).logits.float()
                    forward_differences.append((before - after).abs().max().item())
                    for name, value in effective_weights(model):
                        torch.testing.assert_close(
                            value, before_weights[name], rtol=1e-5, atol=1e-6
                        )
                assert all(s["step"].item() == 4 for s in optimizer.state.values())
                assert scheduler.last_epoch == 5
                assert identities == {n: id(p) for n, p in model.named_parameters()}

        perform(1)
        attach()
        for step in (2, 3, 4):
            perform(step)
        checkpoint = args.output / "checkpoint"
        saved_history = {name: [v.cpu() for v in history] for name, history in histories.items()}
        strategy.save_checkpoint(
            model=wrapped,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_dir=str(checkpoint),
            node_local_rank=int(os.environ["LOCAL_RANK"]),
            client_state={"prepared": {name: s.state_dict() for name, s in observers.items()}},
        )
        perform(5, refresh=True)
        expected_weights = {n: v.clone() for n, v in effective_weights(model)}
        expected_states = {
            name: {key: full_tensor(value).clone() for key, value in optimizer.state[param].items()}
            for name, param in model.named_parameters()
            if param in optimizer.state
        }
        _, loaded = strategy.load_checkpoint(
            model=wrapped, optimizer=optimizer, scheduler=scheduler, ckpt_dir=str(checkpoint)
        )
        for name, s in loaded["client_state"]["prepared"].items():
            observers[name] = PreparedBasisMoments.from_state_dict(s, device="cuda")
            hooks[name].observer = observers[name]
            histories[name] = [v.cuda() for v in saved_history[name]]
            assert observers[name].count == 3
        perform(5, refresh=True)
        for name, value in effective_weights(model):
            torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
        for name, param in model.named_parameters():
            if name in expected_states:
                for key, value in optimizer.state[param].items():
                    torch.testing.assert_close(
                        full_tensor(value), expected_states[name][key], rtol=0, atol=0
                    )
        for hook in hooks.values():
            hook.close()
        for handle in oracle_handles:
            handle.remove()
        observers.clear()
        perform(6)
        assert all(s["step"].item() == 5 for s in optimizer.state.values())
        assert scheduler.last_epoch == 6 and applied_lrs == [0.001] * 7
        assert max(warm_moment_errors) < 0.001, max(warm_moment_errors)
        assert max(gradient_projection_errors) < 0.05, max(gradient_projection_errors)
        assert max(forward_differences) < 0.02, max(forward_differences)
        report = {
            "success": True,
            "world_size": dist.get_world_size(),
            "layers": len(layers),
            "optimizer": type(optimizer).__name__,
            "global_scheduler_step": 6,
            "native_adam_step_after_final_update": 5,
            "prepared_observations_at_refresh": 4,
            "max_relative_native_gradient_projection_error": max(gradient_projection_errors),
            "max_relative_observed_moment_oracle_error": max(warm_moment_errors),
            "max_boundary_logit_abs_difference": max(forward_differences),
            "checkpoint_replay_exact": True,
            "constant_lrs": applied_lrs,
            "clipping_multipliers": clipping,
            "native_parameter_identities_preserved": True,
            "limitations": "Tiny native SkyRL FSDP2 Qwen fixture with gradient checkpointing, native AdamW, distinct rank-local microbatches and a separate dense oracle. Not the production RL worker, full-scale peak VRAM, an inference-engine sync test or performance evidence. BF16 gradient agreement is approximate; the collector uses FP32 projected arithmetic.",
        }
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, report)
        if dist.get_rank() == 0:
            (args.output / "result.json").write_text(
                json.dumps({"ranks": reports}, indent=2) + "\n"
            )
            print(json.dumps(report), flush=True)
        dist.barrier()
    finally:
        for hook in hooks.values():
            hook.close()
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
