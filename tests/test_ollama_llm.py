from types import SimpleNamespace

import pytest

from agentic_ai.llm import LLMError, OllamaLLM
from agentic_ai.schemas import Plan


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(message=SimpleNamespace(content=out))


def make(outputs, retries=2):
    llm = OllamaLLM("test-model", json_retries=retries)
    llm._client = FakeClient(outputs)
    return llm


def test_structured_output_passes_schema():
    llm = make(['{"steps": ["a", "b"]}'])
    assert llm.complete_json([{"role": "user", "content": "x"}], Plan).steps == ["a", "b"]
    assert llm._client.calls[0]["format"]["properties"]["steps"]["type"] == "array"


def test_self_corrects_after_invalid_json():
    llm = make(["oops not json", '{"steps": ["fixed"]}'])
    assert llm.complete_json([{"role": "user", "content": "x"}], Plan).steps == ["fixed"]
    retry_msgs = llm._client.calls[1]["messages"]
    assert retry_msgs[-1]["role"] == "user" and "invalid" in retry_msgs[-1]["content"]


def test_gives_up_after_retries():
    llm = make(["bad", "bad"], retries=1)
    with pytest.raises(LLMError, match="Plan"):
        llm.complete_json([{"role": "user", "content": "x"}], Plan)


def test_connection_error_is_helpful():
    llm = make([ConnectionError("refused")])
    with pytest.raises(LLMError, match="ollama pull test-model"):
        llm.complete([{"role": "user", "content": "x"}])
