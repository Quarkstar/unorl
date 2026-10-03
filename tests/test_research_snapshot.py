import json

import pytest

from scripts.research.snapshot import parse_jsonl


def test_jsonl_keeps_unicode_separators_inside_generated_text():
    rows = [{"response": "a\u2028b\u0085c\u2029d", "score": 1}, {"score": -1}]
    raw = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    assert parse_jsonl(raw) == rows


def test_jsonl_does_not_accept_incomplete_evaluation_record():
    with pytest.raises(json.JSONDecodeError):
        parse_jsonl(b'{"score": 1}\n{"response": "unfinished')
