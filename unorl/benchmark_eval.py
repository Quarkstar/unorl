"""Math benchmark environment with bounded symbolic verification."""

import skyrl_gym
from math_verify import LatexExtractionConfig, parse, verify
from skyrl_gym.envs.aime.env import AIMEEnv
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput

from unorl.grading import score_answer


def symbolic_score(response, ground_truth):
    result = score_answer(response, ground_truth)
    if result["acc"] or result["pred"] is None:
        return result
    # Grade the final explicit answer only, with bounded symbolic parsing.
    gold = parse(
        "$" + str(ground_truth).strip("$") + "$",
        extraction_config=[LatexExtractionConfig()],
        parsing_timeout=2,
    )
    pred = parse(
        "$" + result["pred"].strip("$") + "$",
        extraction_config=[LatexExtractionConfig()],
        parsing_timeout=2,
    )
    correct = bool(verify(gold, pred, timeout_seconds=2))
    return dict(score=1.0 if correct else -1.0, acc=correct, pred=result["pred"])


class BenchmarkEnv(AIMEEnv):
    def step(self, action):
        result = symbolic_score(action, self.ground_truth)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=result["score"],
            done=True,
            metadata=dict(acc=result["acc"], pred=result["pred"]),
        )


skyrl_gym.register("benchmark_math", entry_point="unorl.benchmark_eval:BenchmarkEnv")
