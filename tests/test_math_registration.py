"""Regression test for math environment registration inside Ray workers."""

import subprocess
import sys

import skyrl_gym

from unorl.train import register_math_environments


def test_registration_recreates_missing_specs():
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


def test_preregistered_specs_can_load_environment_classes_in_fresh_process():
    code = """
import skyrl_gym
from skyrl_gym.envs.registration import load_env_creator

targets = {
    "unorl_math": "unorl.grading:MathAnswerEnv",
    "benchmark_math": "unorl.benchmark_eval:BenchmarkEnv",
}
for name, target in targets.items():
    skyrl_gym.register(name, entry_point=target)
for name, target in targets.items():
    creator = load_env_creator(target)
    assert creator.__name__ == target.split(":")[1]
    assert skyrl_gym.registry[name].entry_point == target
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
