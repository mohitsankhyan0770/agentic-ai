"""LLM abstraction.

Every component talks to the model through the small `LLM` protocol below, so the
agent can be unit-tested with a scripted fake and pointed at any backend.
`OllamaLLM` uses Ollama's *structured outputs* (a JSON schema passed via
`format=`) so the model is constrained to emit valid JSON instead of us hoping
a regex can dig it out of free text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the model cannot produce a usable response."""


class LLM(Protocol):
    def complete(self, messages: list[Message]) -> str: ...

    def complete_json(self, messages: list[Message], schema: type[T]) -> T: ...


def extract_json(text: str) -> Any:
    """Parse JSON from model output, tolerating code fences and surrounding prose.

    Uses a decoder scan instead of a regex so nested objects work.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise LLMError(f"No JSON found in model output: {text[:200]!r}")


def parse_into(text: str, schema: type[T]) -> T:
    return schema.model_validate(extract_json(text))


class OllamaLLM:
    """Chat + structured output against a local Ollama server."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        json_retries: int = 2,
    ) -> None:
        import ollama  # imported lazily so tests don't need a server

        self.model = model
        self.temperature = temperature
        self.json_retries = json_retries
        self._client = ollama.Client(host=host)

    def _chat(self, messages: list[Message], fmt: Any = None) -> str:
        try:
            resp = self._client.chat(
                model=self.model,
                messages=messages,
                format=fmt,
                options={"temperature": self.temperature},
            )
        except Exception as exc:  # connection refused, model not pulled, ...
            raise LLMError(
                f"Ollama call failed ({exc}). Is `ollama serve` running and is "
                f"'{self.model}' pulled? Try: ollama pull {self.model}"
            ) from exc
        return resp.message.content or ""

    def complete(self, messages: list[Message]) -> str:
        return self._chat(messages).strip()

    def complete_json(self, messages: list[Message], schema: type[T]) -> T:
        msgs = list(messages)
        last_error: Exception | None = None
        for _ in range(self.json_retries + 1):
            raw = self._chat(msgs, fmt=schema.model_json_schema())
            try:
                return parse_into(raw, schema)
            except (LLMError, ValidationError) as exc:
                last_error = exc
                # Feed the error back so the model can self-correct.
                msgs = msgs + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That was invalid: {exc}. Reply with valid JSON only."},
                ]
        raise LLMError(f"Model failed to return valid {schema.__name__}: {last_error}")
