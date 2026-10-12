"""Math benchmark environment with symbolic verification."""

import skyrl_gym
from math_verify import LatexExtractionConfig, parse, verify
from skyrl_gym.envs.aime.env import AIMEEnv
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput

from unorl.grading import score_answer


def symbolic_score(response, ground_truth):
    result = score_answer(response, ground_truth)
    if result["acc"] or result["pred"] is None:
        return result
    # Grade the final explicit answer only with symbolic parsing. Timeouts are
    # disabled: math_verify implements them with signal.alarm, which raises in a
    # non-main thread (SkyRL runs env steps in a worker pool when
    # environment.skyrl_gym.max_env_workers > 0). The rule-based check above
    # already handles the common case; this fallback only runs when it fails.
    gold = parse(
        "$" + str(ground_truth).strip("$") + "$",
        extraction_config=[LatexExtractionConfig()],
        parsing_timeout=None,
    )
    pred = parse(
        "$" + result["pred"].strip("$") + "$",
        extraction_config=[LatexExtractionConfig()],
        parsing_timeout=None,
    )
    correct = bool(verify(gold, pred, timeout_seconds=None))
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
