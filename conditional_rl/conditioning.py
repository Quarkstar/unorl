"""Small, independently testable conditioning operations."""
import copy
import math

CORRECT = '<|CORRECT|>'
INCORRECT = '<|INCORRECT|>'


def control_ids(tokenizer):
    ids = [tokenizer.convert_tokens_to_ids(token) for token in (CORRECT, INCORRECT)]
    for token, token_id in zip((CORRECT, INCORRECT), ids):
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f'{token} must be a single special token; run scripts/setup.py')
    if len(set(ids)) != 2:
        raise ValueError('Control tokens must have different IDs')
    return ids


def sampling_tokenizer(tokenizer):
    control_ids(tokenizer)
    result = copy.deepcopy(tokenizer)
    if not isinstance(result.chat_template, str):
        raise ValueError('Expected one chat template')
    result.chat_template = "{{ '<|CORRECT|>' }}" + result.chat_template
    return result


def relabel_output(output, tokenizer):
    """Replace the sampled prefix, preserving all response tokens and loss masks."""
    correct_id, incorrect_id = control_ids(tokenizer)
    prompts = output['prompt_token_ids']
    rewards = output['rewards']
    if len(prompts) != len(rewards):
        raise ValueError('Prompt/reward count mismatch')
    result = dict(output)
    result['prompt_token_ids'] = []
    labels = []
    for prompt, reward in zip(prompts, rewards):
        if not prompt or prompt[0] != correct_id:
            raise ValueError('Rollout must start with CORRECT')
        score = float(sum(reward) if isinstance(reward, list) else reward)
        if not math.isfinite(score):
            raise ValueError('Nonfinite reward')
        label = correct_id if score > 0 else incorrect_id
        labels.append(label)
        result['prompt_token_ids'].append([label, *prompt[1:]])
    return result, labels
