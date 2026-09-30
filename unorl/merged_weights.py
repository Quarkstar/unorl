"""Dense W + BA transport and export for adapters accumulated into the base."""

import torch.distributed as dist
from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.weight_sync import WeightChunk
from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker
from transformers import AutoModelForCausalLM

from unorl.low_resource import effective_parameters, effective_weights, full_tensor


class MergedWeightExtractor(fsdp_worker.FSDPWeightExtractor):
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


class BaseWeightExtractor(MergedWeightExtractor):
    """Send accumulated W only; native LoRA synchronization separately sends BA."""

    def extract_weights(self, dtype):
        for name, parameter, _ in effective_parameters(self.model):
            tensor = full_tensor(parameter).to(dtype).contiguous()
            yield WeightChunk(
                names=[name], dtypes=[str(dtype)], shapes=[list(tensor.shape)], tensors=[tensor]
            )


class MergedExportStrategy(FSDPStrategy):
    """Native AdamW/FSDP; export a dense model including all prior merges."""

    def save_hf_model(self, model, output_dir, tokenizer=None, **kwargs):
        import torch

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
