"""Regression test for math environment registration inside Ray workers."""

import skyrl_gym

from unorl.train import register_math_environments


def test_registration_survives_fresh_process_registry():
    original = skyrl_gym.registry.copy()
    try:
        skyrl_gym.registry.clear()
        register_math_environments()
        assert skyrl_gym.registry["benchmark_math"].entry_point == (
            "unorl.benchmark_eval:BenchmarkEnv"
        )
        assert skyrl_gym.registry["unorl_math"].entry_point == "unorl.grading:MathAnswerEnv"
        register_math_environments()
        assert len(skyrl_gym.registry) == 2
    finally:
        skyrl_gym.registry.clear()
        skyrl_gym.registry.update(original)
