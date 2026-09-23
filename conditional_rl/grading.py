"""Shared answer-format handling for conditional SFT and GRPO."""
import copy
import re

import skyrl_gym
from skyrl_gym.envs.aime.env import AIMEEnv
from skyrl_gym.envs.aime.utils import last_boxed_only_string, normalize_final_answer, remove_boxed
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput


def score_answer(response, ground_truth):
    # Select the last explicit answer, never whichever candidate matches the label.
    # Read the complete response so trailing explanations/code cannot hide the answer.
    candidates = [(m.start(), m.group(1)) for m in re.finditer(
        r'(?i)Answer\s*:\s*([^\n<]+)', response)]
    boxed = last_boxed_only_string(response)
    if boxed is not None:
        candidates.append((response.rfind('\\boxed{'), remove_boxed(boxed)))
    prediction = normalize_final_answer(max(candidates, key=lambda item: item[0])[1]) if candidates else None
    correct = prediction is not None and prediction == normalize_final_answer(str(ground_truth))
    return dict(score=1.0 if correct else -1.0, acc=correct, pred=prediction)


def answer_tokenizer(tokenizer):
    result = copy.deepcopy(tokenizer)
    result.chat_template = result.chat_template.replace(
        'Please reason step by step, and put your final answer within ',
        'Please reason step by step. End with Answer: ')
    return result


class MathAnswerEnv(AIMEEnv):
    def step(self, action):
        result = score_answer(action, self.ground_truth)
        return BaseTextEnvStepOutput(observations=[], reward=result['score'], done=True,
                                     metadata=dict(acc=result['acc'], pred=result['pred']))


skyrl_gym.register('conditional_math', entry_point='conditional_rl.grading:MathAnswerEnv')
