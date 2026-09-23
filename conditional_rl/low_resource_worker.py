"""FSDP worker with SGD, periodic LoRA merges, and dense inference exports."""

import re

import ray
import torch
import torch.distributed as dist
from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.distributed.fsdp_utils import should_use_meta_init
from skyrl.backends.skyrl_train.utils.profiler import build_profiler_from_policy_cfg
from skyrl.backends.skyrl_train.weight_sync import WeightChunk
from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from transformers import AutoConfig, AutoModelForCausalLM

from conditional_rl.low_resource import (
    ConstantLearningRate,
    effective_parameters,
    effective_weights,
    make_sgd,
    merge_and_reset,
)


class MergedWeightExtractor(fsdp_worker.FSDPWeightExtractor):
    """Stream one dense parameter at a time, including all accumulated merges."""

    def extract_weights(self, dtype):
        for name, value in effective_weights(self.model):
            tensor = value.to(dtype).contiguous()
            yield WeightChunk(
                names=[name], dtypes=[str(dtype)], shapes=[list(tensor.shape)], tensors=[tensor]
            )

    def get_weight_metadata(self, dtype):
        parameters = list(effective_parameters(self.model))
        return {
            "names": [name for name, _, _ in parameters],
            "dtype_names": [str(dtype).split(".")[-1]] * len(parameters),
            "shapes": [list(parameter.shape) for _, parameter, _ in parameters],
        }


class SGDStrategy(FSDPStrategy):
    """Construct SGD directly; never allocate Adam moment buffers."""

    def _fsdp_init_train_model(self, model, optimizer, scheduler):
        model.model = self._fsdp_init_model(model, is_train=True, is_wrapped=True)
        config = self.optimizer_config
        optimizer = make_sgd(model.model.parameters(), config.lr)
        scheduler = ConstantLearningRate(optimizer)
        return model, optimizer, scheduler

    def save_checkpoint(self, *args, optimizer=None, **kwargs):
        # SkyRL writes empty optimizer and scheduler dictionaries. Full base
        # weights and adapters are still saved for resume.
        if optimizer is not None and optimizer.state:
            raise RuntimeError("Stateless SGD unexpectedly contains optimizer state")
        return super().save_checkpoint(*args, optimizer=None, **kwargs)

    def load_checkpoint(self, *args, optimizer=None, **kwargs):
        result = super().load_checkpoint(*args, optimizer=None, **kwargs)
        scheduler = kwargs.get("scheduler")
        if optimizer is not None and scheduler is not None:
            checkpoint = str(kwargs.get("ckpt_dir", ""))
            match = re.search(r"global_step_(\d+)(?:/|$)", checkpoint)
            if match is None:
                raise ValueError("Resume requires a global_step_N checkpoint path")
            scheduler.last_epoch = int(match.group(1))
        return result

    def save_hf_model(self, model, output_dir, tokenizer=None, **kwargs):
        # PEFT.save_pretrained would export only the latest adapter and lose
        # accumulated base updates. Export an ordinary, fully merged HF model.
        state = {}
        for name, value in effective_weights(model.model):
            if self.is_rank_0():
                state[name] = value.cpu().clone()
        if self.is_rank_0():
            config = self._fix_fsdp_config(model.model.config)
            with torch.device("meta"):
                dense = AutoModelForCausalLM.from_config(config)
            dense.load_state_dict(state, strict=True, assign=True)
            with self._atomic_local_export_dir(output_dir) as work_dir:
                dense.save_pretrained(work_dir, safe_serialization=True, **kwargs)
                if tokenizer is not None:
                    tokenizer.save_pretrained(work_dir)
        dist.barrier()


class LowResourcePolicyWorker(fsdp_worker.FSDPPolicyWorkerBase):
    """Inject training adapters while keeping SkyRL's dense transport contract."""

    def init_model(self, model_path, num_training_steps=None):
        policy = self.cfg.policy
        resource = self.cfg.low_resource
        self.strategy = SGDStrategy(
            fsdp_config=policy.fsdp_config,
            optimizer_config=policy.optimizer_config,
            model_config=policy.model,
            fsdp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        self.strategy.setup_distributed()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if getattr(config, "vision_config", None) is not None:
            raise ValueError("Low-resource worker currently supports text-only models")
        use_meta = should_use_meta_init(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        wrapped = HFModelWrapper(
            model_path,
            use_flash_attention_2=self.cfg.flash_attn,
            bf16=False,
            lora_rank=resource.lora_rank,
            lora_alpha=resource.lora_alpha,
            lora_dropout=0.0,
            lora_init_method="kaiming",
            target_modules="all-linear",
            sequence_parallel_size=policy.sequence_parallel_size,
            remove_microbatch_padding=self.cfg.remove_microbatch_padding,
            use_torch_compile=policy.use_torch_compile,
            model_config_kwargs=policy.model_config_kwargs,
            meta_init=use_meta,
            logprobs_chunk_size=self.cfg.logprobs_chunk_size,
        )
        for parameter in wrapped.model.parameters():
            dtype = (
                torch.float32 if parameter.requires_grad else getattr(torch, resource.base_dtype)
            )
            parameter.data = parameter.data.to(dtype)
        self._seq_parallel_monkey_patch(model=wrapped.model)
        if self.cfg.gradient_checkpointing:
            wrapped.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": self.cfg.gradient_checkpointing_use_reentrant
                }
            )
        self.model, self.optimizer, self.scheduler = self.strategy.prepare((wrapped, None, None))
        self._is_lora = False  # Inference receives dense W + BA, not PEFT files.
        self._is_multimodal_lm_only = False
        self.merge_metrics = {}
        self._set_expandable_segments(True)
        self.profiler = build_profiler_from_policy_cfg(self.cfg)

    async def init_weight_sync_state(self, inference_engine_client, inference_engine_cfg):
        # NCCL initializes only the communication group; metadata is read when
        # sending. Validation excludes transports that bake metadata at init.
        await super().init_weight_sync_state(inference_engine_client, inference_engine_cfg)
        self.weight_extractor = MergedWeightExtractor(self.model.model)

    def optim_step(self):
        norm = super().optim_step()
        self.merge_metrics = {}
        step = self.scheduler.last_epoch
        if norm is not None and not torch.isfinite(torch.tensor(norm)):
            return norm
        if step > 0 and step % self.cfg.low_resource.merge_interval == 0:
            self.merge_metrics = merge_and_reset(
                self.model.model, seed=self.cfg.seed + step * 10000
            )
        if self.optimizer.state:
            raise RuntimeError("SGD must not create persistent optimizer state")
        return norm

    def resource_metrics(self):
        return {
            **self.merge_metrics,
            "optimizer/state_entries": float(len(self.optimizer.state)),
            "relora/completed_cycles": float(
                self.scheduler.last_epoch // self.cfg.low_resource.merge_interval
            ),
        }


PolicyWorker = ray.remote(num_gpus=1)(LowResourcePolicyWorker)
