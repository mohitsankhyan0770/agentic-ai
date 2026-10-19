import pytest

from agentic_ai.llm import LLMError, extract_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('Sure!\n```json\n{"a": {"b": [1, 2]}}\n```') == {"a": {"b": [1, 2]}}


def test_nested_json_in_prose():
    # The original regex r'\{[^{}]*\}' could not handle nesting.
    text = 'Here you go: {"tool": "write_file", "action_input": {"path": "x.txt"}} hope that helps'
    assert extract_json(text)["action_input"]["path"] == "x.txt"


def test_no_json():
    with pytest.raises(LLMError):
        extract_json("no json here")
