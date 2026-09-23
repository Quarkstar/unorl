#!/usr/bin/env python3
"""CPU checks for benchmark answer equivalence and system relabeling."""
import os
import sys
from train import ROOT, PYTHON, environment

if '--worker' not in sys.argv:
    env = environment()
    env['PYTHONPATH'] = str(ROOT / '.deps') + ':' + env['PYTHONPATH']
    os.execve(str(PYTHON), [str(PYTHON), __file__, '--worker'], env)

import json
import asyncio
from math_verify import parse, LatexExtractionConfig
from transformers import AutoTokenizer
from conditional_rl.benchmark_eval import symbolic_score
from conditional_rl.system_conditioning import system_tokenizer, relabel_output
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

rows = [json.loads(line) for line in
        (ROOT / 'data/raw/benchmark-audit/HuggingFaceH4--MATH-500--test.jsonl').read_text().splitlines()]
assert len(rows) == 500
for i, row in enumerate(rows):
    assert parse('$' + row['answer'].strip('$') + '$',
                 extraction_config=[LatexExtractionConfig()], parsing_timeout=2), i
for response, truth, expected in [
    (r'Answer: \frac{2}{4}', r'\frac{1}{2}', True),
    (r'Answer: \sqrt{8}/2', r'\sqrt{2}', True),
    ('Answer: 3\nAnswer: 4', '3', False), ('No answer', '1', False),
]:
    assert symbolic_score(response, truth)['acc'] == expected, response

tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen2.5-Math-1.5B')
prompt = system_tokenizer(tokenizer).apply_chat_template(
    [dict(role='user', content='Compute 1+1.')], tokenize=True,
    add_generation_prompt=True, return_dict=False)
original = dict(prompt_token_ids=[prompt, prompt], rewards=[1, -1],
                response_ids=[[10, 11], [12]], loss_masks=[[1, 1], [1]])
result, labels = relabel_output(original, tokenizer)
assert labels == ['CORRECT', 'INCORRECT']
assert result['response_ids'] == original['response_ids']
assert result['loss_masks'] == original['loss_masks']
assert result['prompt_token_ids'][0] == prompt
assert 'Generate an incorrect solution.' in tokenizer.decode(result['prompt_token_ids'][1])
assert '<|CORRECT|>' not in tokenizer.decode(prompt)

cfg = SkyRLTrainConfig.from_cli_overrides(['environment.skyrl_gym.max_env_workers=0'])
class Engine:
    async def generate(self, batch, **kwargs):
        responses = [r'Answer: \frac{2}{4}', r'Answer: 3']
        ids = [tokenizer.encode(x) for x in responses]
        return dict(responses=responses, response_ids=ids, stop_reasons=['stop'] * 2,
                    response_logprobs=[[-1.] * len(row) for row in ids])

generator = SkyRLGymGenerator(cfg.generator, cfg.environment.skyrl_gym, Engine(), system_tokenizer(tokenizer))
output = asyncio.run(generator.generate_batched(
    [[dict(role='user', content='What is one half?')]] * 2,
    ['benchmark_math'] * 2, [dict(reward_model=dict(ground_truth=r'\frac{1}{2}')) for _ in range(2)], 128))
assert output['rewards'] == [1., -1.]
print('PASS: 500 gold answers parse; symbolic equivalence, final-answer selection, and system relabeling checks passed.')
