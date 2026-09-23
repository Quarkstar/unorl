"""Correctness conditioning using ordinary text in a Qwen system message."""
import copy
import json
import math

FORMAT = 'Please reason step by step. End with Answer: \\boxed{your final answer}.'


def system_text(label=None):
    if label is None:
        return FORMAT
    assert label in ('CORRECT', 'INCORRECT')
    article = 'a' if label == 'CORRECT' else 'an'
    return f'Generate {article} {label.lower()} solution.\n' + FORMAT


def system_tokenizer(tokenizer, label='CORRECT'):
    result = copy.deepcopy(tokenizer)
    # Datasets contain user messages only. This inserts a real system role,
    # using existing vocabulary and the model's original chat serialization.
    result.chat_template = ('{% set messages = [{"role": "system", "content": '
                            + json.dumps(system_text(label)) + '}] + messages %}'
                            + tokenizer.chat_template)
    return result


def prefix_ids(tokenizer, label):
    return tokenizer.encode('<|im_start|>system\n' + system_text(label) + '<|im_end|>\n',
                            add_special_tokens=False)


def relabel_output(output, tokenizer):
    correct = prefix_ids(tokenizer, 'CORRECT')
    result = dict(output, prompt_token_ids=[])
    labels = []
    if len(output['prompt_token_ids']) != len(output['rewards']):
        raise ValueError('Prompt/reward count mismatch')
    for prompt, reward in zip(output['prompt_token_ids'], output['rewards']):
        if prompt[:len(correct)] != correct:
            raise ValueError('Expected correct-solution system prompt')
        score = float(sum(reward) if isinstance(reward, list) else reward)
        if not math.isfinite(score):
            raise ValueError('Nonfinite reward')
        label = 'CORRECT' if score > 0 else 'INCORRECT'
        labels.append(label)
        result['prompt_token_ids'].append(prefix_ids(tokenizer, label) + prompt[len(correct):])
    return result, labels
