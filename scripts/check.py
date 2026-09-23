#!/usr/bin/env python3
"""CPU correctness tests plus configuration validation in the SkyRL environment."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import PYTHON, ROOT, environment

if '--worker' not in sys.argv:
    os.execve(str(PYTHON), [str(PYTHON), __file__, '--worker'], environment())

import asyncio
import copy
import json
import tempfile
import unittest
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils.utils import validate_cfg
from skyrl.backends.skyrl_train.utils.ppo_utils import cross_entropy_loss, ppo_policy_loss

from conditional_rl.conditioning import control_ids, sampling_tokenizer, relabel_output
from conditional_rl.system_conditioning import system_tokenizer, prefix_ids
from conditional_rl.train import (ConditionalTrainer, PositiveOnlyTrainer, MetricsCallback, check_config,
                                  conditional_rollout_metrics, positive_only_output)
from conditional_rl.grading import score_answer, answer_tokenizer


class ConditioningTests(unittest.TestCase):
    def test_conditional_rollout_metrics(self):
        output = {
            'rewards': [1, -1, -1, -1],
            'response_ids': [[1, 2], [1, 2, 3, 4], [1], [1, 2, 3]],
        }
        metrics = conditional_rollout_metrics(
            output, ['CORRECT', 'INCORRECT', 'INCORRECT', 'INCORRECT'], ['a', 'a', 'b', 'b'])
        self.assertEqual(metrics['conditioning/correct_fraction'], 0.25)
        self.assertEqual(metrics['conditioning/incorrect_fraction'], 0.75)
        self.assertEqual(metrics['rollout/prompts_mixed_fraction'], 0.5)
        self.assertEqual(metrics['rollout/prompts_all_incorrect_fraction'], 0.5)
        self.assertEqual(metrics['rollout/prompts_all_correct_fraction'], 0.0)
        self.assertEqual(metrics['rollout/correct_response_tokens_mean'], 2.0)
        self.assertEqual(metrics['rollout/incorrect_response_tokens_mean'], 8 / 3)
        self.assertAlmostEqual(metrics['rollout/reward_std'], 3 ** 0.5 / 2)

    def test_aime_only_evaluation_selection(self):
        from unittest.mock import patch
        from conditional_rl.train import BenchmarkTrainer, RayPPOTrainer
        class Dataset:
            dataframe = [{'data_source': name} for name in ('aime25', 'aime26') for _ in range(30)]
            def __len__(self):
                return len(self.dataframe)
            def __getitem__(self, index):
                return self.dataframe[index]['data_source']
            def collate_fn(self, rows):
                return rows
        trainer = BenchmarkTrainer.__new__(BenchmarkTrainer)
        trainer.cfg = copy.deepcopy(self.cfg)
        trainer.cfg.trainer.export_path = '/tmp/aime-only-check'
        trainer.cfg.trainer.eval_batch_size = 128
        trainer.eval_dataset = Dataset()
        trainer.eval_dataloader = object()
        seen = []
        async def fake_eval(self, scraper=None):
            name = self.eval_dataloader[0][0]
            seen.append((name, len(self.eval_dataloader[0]), self.cfg.generator.eval_n_samples_per_prompt))
            return {f'eval/{name}/pass_at_8': 0.5}
        with patch.object(RayPPOTrainer, 'eval', fake_eval):
            metrics = asyncio.run(trainer.eval())
        self.assertEqual(seen, [('aime25', 30, 8), ('aime26', 30, 8)])
        self.assertEqual(metrics['eval/aime26/pass@8'], 0.5)
        self.assertFalse(any('math500' in key or 'amc23' in key for key in metrics))

    def test_training_benchmark_sample_counts_and_restore(self):
        from unittest.mock import patch
        from conditional_rl.train import BenchmarkTrainer, RayPPOTrainer
        trainer = BenchmarkTrainer.__new__(BenchmarkTrainer)
        trainer.cfg = copy.deepcopy(self.cfg)
        trainer.cfg.trainer.export_path = '/tmp/benchmark-test-exports'
        trainer.eval_dataloader = object()
        original_loader = trainer.eval_dataloader
        trainer.benchmark_batches = {name: [name] for name in ('math500', 'amc23', 'aime25')}
        seen = []
        async def fake_eval(self, scraper=None):
            name = self.eval_dataloader[0]
            seen.append((name, self.cfg.generator.eval_n_samples_per_prompt))
            return {f'eval/{name}/mean_positive_reward': 0.25,
                    f'eval/{name}/pass_at_8': 0.5, 'eval/all/mean_positive_reward': 0.25}
        with patch.object(RayPPOTrainer, 'eval', fake_eval):
            metrics = asyncio.run(trainer.eval())
        self.assertEqual(seen, [('math500', 1), ('amc23', 8), ('aime25', 8)])
        self.assertEqual(metrics['eval/math500/accuracy'], 0.25)
        self.assertEqual(metrics['eval/amc23/pass@8'], 0.5)
        self.assertNotIn('eval/all/mean_positive_reward', metrics)
        self.assertIs(trainer.eval_dataloader, original_loader)
        self.assertEqual(trainer.cfg.generator.eval_n_samples_per_prompt, 8)
        self.assertEqual(trainer.cfg.trainer.export_path, '/tmp/benchmark-test-exports')

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen2.5-Math-1.5B-conditional')
        cls.cfg = SkyRLTrainConfig.from_cli_overrides([
            f'{key}={json.dumps(value)}' for key, value in
            json.loads((ROOT / 'configs/experiment.json').read_text()).items()
        ] + [f'trainer.policy.model.path={ROOT / "models/Qwen2.5-Math-1.5B-conditional"}'])

    def test_dataset_splits_and_eval_context(self):
        import pandas as pd
        train = pd.read_parquet(ROOT / 'data/train.parquet')
        validation = pd.read_parquet(ROOT / 'data/dapo-validation.parquet')
        evaluation = pd.read_parquet(ROOT / 'data/eval.parquet')
        self.assertEqual(len(train), 7500)
        self.assertEqual(len(validation), 500)
        self.assertEqual(evaluation.data_source.value_counts().to_dict(),
                         dict(aime2024=30, aime2025=30, aime2026=30))
        def key(prompt):
            return ''.join(''.join(message['content'].split()) for message in prompt)
        train_keys = set(train.prompt.map(key))
        published = pd.read_parquet(ROOT / 'data/raw/dapo-stratified/data/train.parquet')
        self.assertEqual(train.prompt.map(key).tolist(), published.prompt.map(key).tolist())
        self.assertFalse(train_keys & set(evaluation.prompt.map(key)))
        for prompt in evaluation.prompt:
            tokens = self.tokenizer.apply_chat_template(prompt.tolist(), add_generation_prompt=True,
                                                        tokenize=True, return_dict=False)
            self.assertLessEqual(len(tokens), self.cfg.trainer.max_prompt_length)

    def test_avg_at_8_reporting(self):
        from skyrl.train.generators.utils import get_metrics_from_generator_output
        # One correct response among eight means avg@8=1/8, whereas pass@8=1.
        output = dict(rewards=[1.] + [-1.] * 7)
        metrics = get_metrics_from_generator_output(output, ['question'] * 8)
        self.assertEqual(metrics['mean_positive_reward'], 0.125)
        self.assertEqual(metrics['pass_at_n'], 1.0)
        with tempfile.TemporaryDirectory() as folder:
            cfg = copy.deepcopy(self.cfg)
            cfg.trainer.export_path = folder + '/exports'
            callback_input = SimpleNamespace(global_step=0, metrics={
                'eval/aime2025/mean_positive_reward': metrics['mean_positive_reward']})
            MetricsCallback().on_eval_end(SimpleNamespace(cfg=cfg), callback_input, None)
            self.assertEqual(callback_input.metrics['eval/aime2025/avg@8'], 0.125)

    def test_shared_answer_formats(self):
        for response in [r'Answer: 42', r'Final result: \boxed{42}',
                         r'Answer: \boxed{42}', r'Final result: \boxed{42}' + '\n' + 'code ' * 100]:
            self.assertEqual(score_answer(response, '42')['score'], 1.0)
        for response in ['', 'The intermediate number is 42.', r'Answer: 41',
                         r'\boxed{42}' + '\nAnswer: 41', r'Answer: 42' + '\n' + r'\boxed{41}']:
            self.assertEqual(score_answer(response, '42')['score'], -1.0)
        plain = answer_tokenizer(self.tokenizer)
        prompt = [{'role': 'user', 'content': 'Compute 6*7.'}]
        text = plain.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        self.assertIn(r'End with Answer: \boxed{}.', text)
        conditioned = sampling_tokenizer(plain).apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        self.assertEqual(conditioned, '<|CORRECT|>' + text)

    def test_config(self):
        check_config(self.cfg)
        validate_cfg(self.cfg)

    def test_prefix_and_relabel(self):
        tok = sampling_tokenizer(self.tokenizer)
        prompts = [[{'role': 'user', 'content': 'Compute 1+1.'}]] * 2
        ids = tok.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, return_dict=False)
        original = self.tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, return_dict=False)
        correct, incorrect = control_ids(tok)
        self.assertEqual(ids, [[correct, *prompt] for prompt in original])
        output = dict(prompt_token_ids=ids, response_ids=[[10, 11], [12]],
                      rewards=[[0, 1], [-1]], loss_masks=[[1, 1], [1]])
        before = copy.deepcopy(output)
        relabeled, labels = relabel_output(output, tok)
        self.assertEqual(labels, [correct, incorrect])
        self.assertEqual(output, before)
        self.assertEqual(relabeled['response_ids'], output['response_ids'])
        self.assertEqual(relabeled['loss_masks'], output['loss_masks'])
        self.assertEqual(relabeled['prompt_token_ids'][1][1:], ids[1][1:])

    def test_positive_only_masking_preserves_prompts_and_input(self):
        output = dict(
            prompt_token_ids=[[1, 2], [1, 3]],
            response_ids=[[10, 11], [12, 13, 14]],
            rewards=[[0, 1], [0, 0, -1]],
            loss_masks=[[1, 1], [1, 0, 1]],
        )
        before = copy.deepcopy(output)
        filtered, labels = positive_only_output(output)
        self.assertEqual(labels, ['CORRECT', 'INCORRECT'])
        self.assertEqual(filtered['prompt_token_ids'], output['prompt_token_ids'])
        self.assertEqual(filtered['loss_masks'], [[1, 1], [0, 0, 0]])
        self.assertEqual(output, before)

    def test_standard_generator_reward_and_training_batch(self):
        tok = system_tokenizer(self.tokenizer)
        correct, incorrect = [prefix_ids(tok, label) for label in ('CORRECT', 'INCORRECT')]
        class Engine:
            async def generate(self, batch, **kwargs):
                assert all(prompt[:len(correct)] == correct for prompt in batch['prompt_token_ids'])
                responses = [r'Final result: \boxed{2}' + '\n' + 'Trailing explanation. ' * 30, r'Answer: \boxed{3}']
                ids = [tok.encode(response, add_special_tokens=False) for response in responses]
                return dict(responses=responses, response_ids=ids, stop_reasons=['stop'] * 2,
                            response_logprobs=[[-1.0] * len(row) for row in ids])
        generator = SkyRLGymGenerator(self.cfg.generator, self.cfg.environment.skyrl_gym, Engine(), tok)
        prompts = [[dict(role='user', content='Compute 1+1.')]] * 2
        extras = [dict(reward_model=dict(ground_truth='2')) for _ in range(2)]
        output = asyncio.run(generator.generate_batched(prompts, ['conditional_math'] * 2, extras, 128))
        self.assertEqual(output['rewards'], [1.0, -1.0])
        trainer = ConditionalTrainer.__new__(ConditionalTrainer)
        trainer.tokenizer = self.tokenizer
        trainer.cfg = copy.deepcopy(self.cfg)
        trainer.cfg.trainer.train_batch_size = 1
        trainer.cfg.trainer.policy_mini_batch_size = 1
        trainer.cfg.generator.n_samples_per_prompt = 2
        trainer.all_metrics = {}
        trainer.global_step = 1
        trainer.dispatch = SimpleNamespace(get_lcm_dp_size=lambda: 1)
        with tempfile.TemporaryDirectory() as directory:
            trainer.cfg.trainer.export_path = directory + '/exports'
            output, uids = trainer.postprocess_generator_output(output, ['0', '0'])
            batch = trainer.convert_to_training_input(output, uids)
        for row, label in enumerate([correct, incorrect]):
            attended = batch['sequences'][row][batch['attention_mask'][row].bool()]
            self.assertEqual(attended[:len(label)].tolist(), label)
        batch = trainer.compute_advantages_and_returns(batch)
        batch = trainer._normalize_advantages(batch, [(0, 2)])
        self.assertAlmostEqual(batch['loss_mask'].sum().item(), 1.0, places=6)
        self.assertTrue((batch['advantages'] > 0).all())

    def test_positive_only_training_batch_uses_only_accepted_response(self):
        trainer = PositiveOnlyTrainer.__new__(PositiveOnlyTrainer)
        trainer.tokenizer = self.tokenizer
        trainer.cfg = copy.deepcopy(self.cfg)
        trainer.cfg.trainer.train_batch_size = 1
        trainer.cfg.trainer.policy_mini_batch_size = 1
        trainer.cfg.generator.n_samples_per_prompt = 2
        trainer.all_metrics = {}
        trainer.global_step = 1
        trainer.dispatch = SimpleNamespace(get_lcm_dp_size=lambda: 1)
        prompt = self.tokenizer.apply_chat_template(
            [dict(role='system', content='Generate a correct solution.'),
             dict(role='user', content='Compute 1+1.')],
            tokenize=True, add_generation_prompt=True, return_dict=False)
        output = dict(
            prompt_token_ids=[prompt, prompt],
            response_ids=[[10, 11], [12, 13, 14]],
            rewards=[[0, 1], [0, 0, -1]],
            loss_masks=[[1, 1], [1, 1, 1]],
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer.cfg.trainer.export_path = directory + '/exports'
            batch = trainer.convert_to_training_input(output, ['0', '0'])
            audit = json.loads((Path(directory) / 'positive-only.jsonl').read_text())
        batch = trainer.compute_advantages_and_returns(batch)
        batch = trainer._normalize_advantages(batch, [(0, 2)])
        self.assertAlmostEqual(batch['loss_mask'][0].sum().item(), 1.0)
        self.assertEqual(batch['loss_mask'][1].sum().item(), 0.0)
        self.assertEqual(audit['accepted'], 1)
        self.assertEqual(audit['rejected'], 1)
        self.assertEqual(trainer.all_metrics['rejection/acceptance_rate'], 0.5)

    def test_grpo_retains_signed_advantages(self):
        config = copy.deepcopy(self.cfg.trainer.algorithm)
        config.policy_loss_type = 'regular'
        log_probs = torch.tensor([[-1.], [-1.]], requires_grad=True)
        advantages = torch.tensor([[1.], [-1.]])
        loss, _ = ppo_policy_loss(log_probs, log_probs.detach().clone(), advantages,
                                  config, torch.ones_like(log_probs))
        loss.backward()
        self.assertLess(log_probs.grad[0].item(), 0)
        self.assertGreater(log_probs.grad[1].item(), 0)

    def test_positive_loss_masking_and_microbatch_equivalence(self):
        log_probs = torch.tensor([[-1., -2., -8.], [-3., -4., -9.]], requires_grad=True)
        mask = torch.tensor([[1., 1., 0.], [1., 1., 0.]]) / 4
        negative_advantages = -torch.ones_like(log_probs)
        loss, _ = cross_entropy_loss(log_probs, None, negative_advantages, self.cfg.trainer.algorithm, mask)
        self.assertAlmostEqual(loss.item(), 2.5)
        loss.backward()
        self.assertTrue(torch.equal(log_probs.grad, -mask))
        # Both reward groups get the same positive imitation gradient; padding gets none.
        split_loss = sum(cross_entropy_loss(log_probs[i:i+1], None, None,
                          self.cfg.trainer.algorithm, mask[i:i+1])[0] for i in range(2))
        self.assertAlmostEqual(split_loss.item(), loss.item())


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)
