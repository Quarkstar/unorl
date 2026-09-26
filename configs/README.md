# Training profiles

The current controlled comparison uses:

- `qwen3-4b-base-grpo-lora-r1-blog.json`: native rank-1 LoRA, AdamW, 8 rollouts.
- `qwen3-4b-base-reinforce-adamw-r1.json`: raw signed REINFORCE, AdamW, 1 rollout.
- `qwen3-4b-base-reinforce-batchnorm-adamw-r1.json`: normalized REINFORCE, AdamW, 1 rollout.
- `qwen3-4b-base-ppo-lora-r1-valuewarmup.json`: PPO with critic warmup.

Other profiles preserve earlier low-resource settings for CPU checks and method
research. Profiles containing `trainer.low_resource.*` require the custom
`unorl.low_resource_train` entrypoint and use SGD; they must not be mistaken for the
current AdamW comparison. Smoke *configurations* remain as reusable test inputs;
smoke run records have been removed.

Asset paths are relative to `UNORL_PROJECT_ROOT`. The launcher converts them to
absolute paths and stores the resolved configuration in the run directory.
