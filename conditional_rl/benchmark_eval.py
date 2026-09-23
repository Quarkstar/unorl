"""Evaluation-only SkyRL runner; one set of eight engines for all comparisons."""
import asyncio
import json
import os
import sys
from pathlib import Path

import ray
import skyrl_gym
from math_verify import parse, verify, LatexExtractionConfig
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.evaluate import evaluate
from skyrl.train.utils.trainer_utils import build_dataloader
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg
from skyrl_gym.envs.aime.env import AIMEEnv
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput

from conditional_rl.grading import score_answer
from conditional_rl.system_conditioning import system_tokenizer


def symbolic_score(response, ground_truth):
    result = score_answer(response, ground_truth)
    if result['acc'] or result['pred'] is None:
        return result
    # Grade the final explicit answer only, with bounded symbolic parsing.
    gold = parse('$' + str(ground_truth).strip('$') + '$',
                 extraction_config=[LatexExtractionConfig()], parsing_timeout=2)
    pred = parse('$' + result['pred'].strip('$') + '$',
                 extraction_config=[LatexExtractionConfig()], parsing_timeout=2)
    correct = bool(verify(gold, pred, timeout_seconds=2))
    return dict(score=1.0 if correct else -1.0, acc=correct, pred=result['pred'])


class BenchmarkEnv(AIMEEnv):
    def step(self, action):
        result = symbolic_score(action, self.ground_truth)
        return BaseTextEnvStepOutput(observations=[], reward=result['score'], done=True,
                                    metadata=dict(acc=result['acc'], pred=result['pred']))


skyrl_gym.register('benchmark_math', entry_point='conditional_rl.benchmark_eval:BenchmarkEnv')


class BenchmarkEval(EvalOnlyEntrypoint):
    async def run(self, client):
        root = Path(os.environ['CONDITIONAL_PROJECT_ROOT'])
        run = Path(self.cfg.trainer.export_path).parent
        results = []
        try:
            await client.wake_up()
            # Keep identical base weights, sampling settings, and questions.
            for mode in ('baseline', 'conditional_system'):
                token = system_tokenizer(self.tokenizer, None if mode == 'baseline' else 'CORRECT')
                generator = self.get_generator(self.cfg, token, client)
                for benchmark, count in [('math500', 1), ('amc23', 8), ('aime25', 8)]:
                    self.cfg.data.val_data = [str(root / f'data/{benchmark}-benchmark.parquet')]
                    self.cfg.generator.eval_n_samples_per_prompt = count
                    self.cfg.trainer.export_path = str(run / mode / benchmark)
                    dataset = self.get_eval_dataset()
                    expected = dict(math500=500, amc23=40, aime25=30)[benchmark]
                    assert len(dataset) == expected, (benchmark, len(dataset))
                    print(f'START {mode} {benchmark} questions={expected} samples={count}', flush=True)
                    metrics = await evaluate(
                        eval_dataloader=build_dataloader(self.cfg, dataset, is_train=False),
                        generator=generator, cfg=self.cfg, global_step=None, tokenizer=token)
                    key = 'mean_positive_reward' if count == 1 else 'pass_at_8'
                    result = dict(mode=mode, benchmark=benchmark, questions=expected,
                                  samples_per_question=count, metric='accuracy' if count == 1 else 'pass@8',
                                  value=metrics[f'eval/all/{key}'], metrics=metrics)
                    results.append(result)
                    (run / 'results.json').write_text(json.dumps(results, indent=2))
                    print('RESULT ' + json.dumps(result), flush=True)
            return results
        finally:
            await client.aclose()


@ray.remote(num_cpus=1)
def entrypoint(cfg):
    # Ray serializes __main__ functions by value; explicitly import the module
    # in the worker so the custom environment registration executes there.
    from conditional_rl.benchmark_eval import BenchmarkEval as RegisteredBenchmarkEval
    experiment = RegisteredBenchmarkEval(cfg)
    client = experiment.get_inference_client()
    return asyncio.run(experiment.run(client))


def main():
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    try:
        ray.get(entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
