"""SkyRL extension: standard math rollouts plus reward-conditioned positive SFT."""
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import ray
import torch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.utils import validate_cfg
from skyrl.train.utils.callbacks import TrainingCallback

from conditional_rl.system_conditioning import relabel_output, system_tokenizer
from conditional_rl.grading import answer_tokenizer


def positive_only_output(generator_output):
    """Keep on-policy positive responses as CE targets and mask rejected ones."""
    rewards = generator_output['rewards']
    masks = generator_output['loss_masks']
    if len(rewards) != len(masks):
        raise ValueError('Reward/loss-mask count mismatch')
    output = dict(generator_output, loss_masks=[])
    labels = []
    for reward, mask in zip(rewards, masks):
        score = float(sum(reward) if isinstance(reward, list) else reward)
        if not math.isfinite(score):
            raise ValueError('Nonfinite reward')
        accepted = score > 0
        labels.append('CORRECT' if accepted else 'INCORRECT')
        output['loss_masks'].append(list(mask) if accepted else [0] * len(mask))
    return output, labels


def conditional_rollout_metrics(generator_output, labels, uids):
    """Metrics that describe reward/label balance before the CE update."""
    rewards = []
    lengths = defaultdict(list)
    outcomes = defaultdict(list)
    response_ids = generator_output.get('response_ids', [[] for _ in labels])
    if not (len(labels) == len(uids) == len(response_ids) == len(generator_output['rewards'])):
        raise ValueError('Conditional metric inputs have inconsistent lengths')
    for reward, label, uid, response in zip(generator_output['rewards'], labels, uids, response_ids):
        score = float(sum(reward) if isinstance(reward, list) else reward)
        rewards.append(score)
        lengths[label.lower()].append(len(response))
        outcomes[uid].append(score > 0)
    prompt_states = [set(values) for values in outcomes.values()]
    count = len(labels)
    mean_reward = sum(rewards) / count
    metrics = {
        'conditioning/correct_fraction': labels.count('CORRECT') / count,
        'conditioning/incorrect_fraction': labels.count('INCORRECT') / count,
        'rollout/reward_std': math.sqrt(sum((value - mean_reward) ** 2 for value in rewards) / count),
        'rollout/reward_min': min(rewards),
        'rollout/reward_max': max(rewards),
        'rollout/prompts_all_correct_fraction': sum(state == {True} for state in prompt_states) / len(prompt_states),
        'rollout/prompts_all_incorrect_fraction': sum(state == {False} for state in prompt_states) / len(prompt_states),
        'rollout/prompts_mixed_fraction': sum(state == {False, True} for state in prompt_states) / len(prompt_states),
    }
    for label in ('correct', 'incorrect'):
        values = lengths[label]
        metrics[f'rollout/{label}_response_tokens_mean'] = sum(values) / len(values) if values else 0.0
    return metrics


class MetricsCallback(TrainingCallback):
    def on_log(self, trainer, callback_input, control):
        self.write(trainer, 'metrics.jsonl', callback_input.global_step, callback_input.logs)

    def on_eval_end(self, trainer, callback_input, control):
        if not hasattr(trainer, 'benchmark_batches') and trainer.cfg.generator.eval_n_samples_per_prompt == 8:
            for key, value in list(callback_input.metrics.items()):
                if key.endswith('/mean_positive_reward'):
                    callback_input.metrics[key.replace('/mean_positive_reward', '/avg@8')] = value
        self.write(trainer, 'evaluation.jsonl', callback_input.global_step, callback_input.metrics)

    @staticmethod
    def write(trainer, filename, step, metrics):
        path = Path(trainer.cfg.trainer.export_path).parent / filename
        with path.open('a') as handle:
            handle.write(json.dumps(dict(step=step, metrics=metrics), default=float) + '\n')


class BenchmarkTrainer(RayPPOTrainer):
    async def eval(self, vllm_metrics_scraper=None):
        from conditional_rl.benchmark_eval import BenchmarkEnv  # register in Ray worker
        if not hasattr(self, 'benchmark_batches'):
            self.benchmark_batches = {}
            expected_sizes = {'math500': 500, 'amc23': 40, 'aime25': 30, 'aime26': 30}
            selected = {self.eval_dataset.dataframe[i]['data_source'] for i in range(len(self.eval_dataset))}
            assert selected and selected <= expected_sizes.keys(), selected
            for name, expected in expected_sizes.items():
                if name not in selected:
                    continue
                rows = [self.eval_dataset[i] for i in range(len(self.eval_dataset))
                        if self.eval_dataset.dataframe[i]['data_source'] == name]
                assert len(rows) == expected, (name, len(rows))
                size = self.cfg.trainer.eval_batch_size
                self.benchmark_batches[name] = [self.eval_dataset.collate_fn(rows[i:i+size])
                                                for i in range(0, len(rows), size)]
        old_loader = self.eval_dataloader
        n_samples = self.cfg.generator.eval_n_samples_per_prompt
        old_export = self.cfg.trainer.export_path
        metrics = {}
        try:
            for name, batches in self.benchmark_batches.items():
                self.eval_dataloader = batches
                self.cfg.trainer.export_path = str(Path(old_export) / name)
                current = await super().eval(vllm_metrics_scraper)
                metrics.update({k: v for k, v in current.items() if k.startswith(f'eval/{name}/')})
                if name == 'math500':
                    metrics[f'eval/{name}/accuracy'] = current[f'eval/{name}/mean_positive_reward']
                else:
                    metrics[f'eval/{name}/avg@{n_samples}'] = current[
                        f'eval/{name}/mean_positive_reward'
                    ]
                    metrics[f'eval/{name}/pass@{n_samples}'] = current[
                        f'eval/{name}/pass_at_{n_samples}'
                    ]
        finally:
            self.eval_dataloader = old_loader
            self.cfg.trainer.export_path = old_export
        return metrics

    async def train(self):
        try:
            await super().train()
        finally:
            await self.inference_engine_client.aclose()


class BaselineExperiment(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return super().get_generator(cfg, system_tokenizer(tokenizer, None), inference_engine_client)

    def get_trainer(self, **kwargs):
        trainer = BenchmarkTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


class ConditionalTrainer(BenchmarkTrainer):
    def convert_to_training_input(self, generator_output, uids):
        output, labels = relabel_output(generator_output, self.tokenizer)
        counts = dict(correct=labels.count('CORRECT'), incorrect=labels.count('INCORRECT'))
        self.all_metrics.update({f'conditioning/{key}': value for key, value in counts.items()})
        self.all_metrics.update(conditional_rollout_metrics(generator_output, labels, uids))
        audit = Path(self.cfg.trainer.export_path).parent / 'conditioning.jsonl'
        audit.parent.mkdir(parents=True, exist_ok=True)
        with audit.open('a') as handle:
            handle.write(json.dumps(dict(step=self.global_step, **counts,
                                         sampled_system_label='CORRECT', objective='positive_cross_entropy')) + '\n')
        return super().convert_to_training_input(output, uids)

    def compute_advantages_and_returns(self, data):
        # The built-in cross_entropy loss ignores advantages. Supply positive placeholders
        # to satisfy the trainer batch contract, without any group filtering or RL weights.
        data['advantages'] = torch.ones_like(data['rewards'])
        data['returns'] = torch.zeros_like(data['rewards'])
        return data

    def _normalize_advantages(self, data, mini_batch_boundaries, prompt_boundaries=None):
        # RL puts normalization in advantages; CE ignores them. Match SkyRL's
        # SFT collator by scaling response masks over each global minibatch.
        masks = data['loss_mask'].float().clone()
        for start, end in mini_batch_boundaries:
            masks[start:end] /= masks[start:end].sum().clamp_min(1)
        data['loss_mask'] = masks
        return data


class PositiveOnlyTrainer(ConditionalTrainer):
    def convert_to_training_input(self, generator_output, uids):
        output, labels = positive_only_output(generator_output)
        accepted = labels.count('CORRECT')
        rejected = labels.count('INCORRECT')
        total = len(labels)
        self.all_metrics.update({
            'conditioning/correct': accepted,
            'conditioning/incorrect': rejected,
            'rejection/accepted_responses': accepted,
            'rejection/rejected_responses': rejected,
            'rejection/acceptance_rate': accepted / total,
        })
        self.all_metrics.update(conditional_rollout_metrics(generator_output, labels, uids))
        audit = Path(self.cfg.trainer.export_path).parent / 'positive-only.jsonl'
        audit.parent.mkdir(parents=True, exist_ok=True)
        with audit.open('a') as handle:
            handle.write(json.dumps(dict(
                step=self.global_step,
                accepted=accepted,
                rejected=rejected,
                sampled_system_label='CORRECT',
                objective='on_policy_positive_only_cross_entropy',
            )) + '\n')
        # Bypass ConditionalTrainer's prompt relabeling: every rollout and every
        # training sequence keeps the same positive system instruction.
        return BenchmarkTrainer.convert_to_training_input(self, output, uids)


class ConditionalExperiment(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return super().get_generator(cfg, system_tokenizer(tokenizer), inference_engine_client)

    def get_trainer(self, **kwargs):
        trainer = ConditionalTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


class PositiveOnlyExperiment(ConditionalExperiment):
    def get_trainer(self, **kwargs):
        trainer = PositiveOnlyTrainer(**kwargs)
        trainer.add_callback(MetricsCallback())
        return trainer


def check_config(cfg):
    algo = cfg.trainer.algorithm
    assert algo.policy_loss_type == 'cross_entropy'
    assert not algo.use_kl_loss and not algo.use_kl_in_reward
    assert not algo.zero_variance_filter and algo.dynamic_sampling.type is None
    assert algo.off_policy_correction.tis_ratio_type is None
    assert algo.off_policy_correction.sequence_mask_metric is None
    assert cfg.generator.batched and cfg.generator.max_turns == 1
    assert not cfg.generator.step_wise_trajectories
    assert cfg.environment.env_class == 'conditional_math'


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    from conditional_rl.train import BaselineExperiment, ConditionalExperiment, PositiveOnlyExperiment
    from conditional_rl.benchmark_eval import BenchmarkEnv
    experiments = {
        'grpo': BaselineExperiment,
        'conditional': ConditionalExperiment,
        'positive': PositiveOnlyExperiment,
    }
    mode = os.environ.get('SKYRL_CONDITIONAL_MODE', 'conditional')
    if mode not in experiments:
        raise ValueError(f'Unknown SKYRL_CONDITIONAL_MODE: {mode}')
    experiment = experiments[mode]
    experiment(cfg).run()


def main():
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    if os.environ.get('SKYRL_CONDITIONAL_MODE') != 'grpo':
        check_config(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
