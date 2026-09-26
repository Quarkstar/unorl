"""Shared answer-format prompt; serialized identically to the baseline runs."""

import copy
import json

FORMAT = r"Please reason step by step. End with Answer: \boxed{your final answer}."


def system_tokenizer(tokenizer):
    result = copy.deepcopy(tokenizer)
    result.chat_template = (
        '{% set messages = [{"role": "system", "content": '
        + json.dumps(FORMAT)
        + "}] + messages %}"
        + tokenizer.chat_template
    )
    return result
