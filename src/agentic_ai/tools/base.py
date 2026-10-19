"""Tool abstraction: typed arguments, validation, and failure isolation.

A tool never raises into the agent loop. Bad arguments or runtime errors come
back as an error *observation* the model can read and recover from.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError


@dataclass
class ToolResult:
    ok: bool
    output: str
    duration_ms: float = 0.0

    def as_observation(self, limit: int = 2000) -> str:
        text = self.output if len(self.output) <= limit else self.output[:limit] + " …[truncated]"
        return text if self.ok else f"ERROR: {text}"


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "args": schema.get("properties", {}),
            "required": schema.get("required", []),
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self.tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self.tools)

    def describe(self) -> str:
        """Tool catalogue rendered for the system prompt."""
        return "\n".join(json.dumps(t.spec()) for t in self.tools.values())

    def execute(self, name: str, args: dict[str, Any] | None) -> ToolResult:
        start = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(False, f"Unknown tool '{name}'. Available: {', '.join(self.names())}")
        try:
            parsed = tool.args_model.model_validate(args or {})
        except ValidationError as exc:
            errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return ToolResult(False, f"Invalid arguments for {name}: {errors}")
        try:
            output = tool.fn(**parsed.model_dump())
            ok = True
        except Exception as exc:  # tool bugs must not crash the agent
            output, ok = f"{type(exc).__name__}: {exc}", False
        return ToolResult(ok, str(output), (time.perf_counter() - start) * 1000)
