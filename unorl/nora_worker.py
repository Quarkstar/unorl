"""NoRA-init policy worker retaining SkyRL's native FSDP and AdamW path."""

import json
from unittest.mock import patch

import ray
from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

from unorl.nora import normalize_lora_initialization


class NoRAInitModelWrapper(HFModelWrapper):
    def __init__(self, *args, **kwargs):
        if kwargs.get("lora_init_method") != "nora_init":
            raise ValueError("NoRA worker requires lora.init_method=nora_init")
        kwargs["lora_init_method"] = "kaiming"
        super().__init__(*args, **kwargs)
        audit = normalize_lora_initialization(self.model)
        print("NoRA-init before FSDP: " + json.dumps(audit), flush=True)


class NoRAInitPolicyWorker(fsdp_worker.FSDPPolicyWorkerBase):
    def init_model(self, model_path, num_training_steps=None):
        # Actor-local substitution only while native init builds the model.
        # Normalization runs before sharding, optimizer creation and resume.
        with patch.object(fsdp_worker, "HFModelWrapper", NoRAInitModelWrapper):
            return super().init_model(model_path, num_training_steps)


PolicyWorker = ray.remote(num_gpus=1)(NoRAInitPolicyWorker)
