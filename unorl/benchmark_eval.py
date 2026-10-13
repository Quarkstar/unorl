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
    # Grade the final explicit answer only with bounded symbolic parsing.
    # math_verify implements these timeouts with signal.alarm, which only works on
    # the main thread, so evaluation must run with
    # environment.skyrl_gym.max_env_workers = 0 (the default in our configs). A
    # finite timeout is required: with None a pathological response can hang the
    # parser indefinitely (observed 2026-10-13).
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


if "benchmark_math" not in skyrl_gym.registry:
    skyrl_gym.register("benchmark_math", entry_point="unorl.benchmark_eval:BenchmarkEnv")
