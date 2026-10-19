"""Test doubles: a scripted LLM and a deterministic embedder (no Ollama needed)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from agentic_ai.config import Settings
from agentic_ai.llm import LLMError, parse_into
from agentic_ai.tools import default_registry


class ScriptedLLM:
    """Returns pre-written responses in order and records every prompt it saw."""

    model = "scripted"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def _next(self, messages) -> Any:
        self.calls.append(messages)
        if not self.responses:
            raise LLMError("ScriptedLLM ran out of responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def complete(self, messages) -> str:
        return str(self._next(messages))

    def complete_json(self, messages, schema: type[BaseModel]):
        item = self._next(messages)
        return parse_into(item if isinstance(item, str) else json.dumps(item), schema)


class BagOfWordsEmbedder:
    """Tiny deterministic embedder: similar wording → similar vectors."""

    VOCAB_SIZE = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.VOCAB_SIZE
            for word in text.lower().split():
                vec[sum(map(ord, word.strip(".,?!:"))) % self.VOCAB_SIZE] += 1.0
            out.append(vec)
        return out


def tool(name: str, thought: str = "use tool", **args) -> dict:
    return {"thought": thought, "action": name, "action_input": args}


def final(answer: str) -> dict:
    return {"thought": "done", "action": "final_answer", "action_input": {}, "answer": answer}


@pytest.fixture
def registry(tmp_path):
    return default_registry(tmp_path / "ws", enable_web=False)


@pytest.fixture
def settings(tmp_path):
    return Settings(workspace_dir=tmp_path / "ws", memory_dir=tmp_path / "mem", trace_dir=tmp_path / "runs")
