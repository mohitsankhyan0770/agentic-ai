"""Typed contracts between the model and the code.

These pydantic models double as the JSON schemas passed to the LLM, so the
same definition constrains generation *and* validates the result.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


class AgentDecision(BaseModel):
    """One ReAct step: reason, then either call a tool or finish."""

    thought: str = Field(..., description="Brief reasoning about what to do next")
    action: str = Field(..., description="A tool name, or 'final_answer' when done")
    action_input: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    answer: str | None = Field(None, description="The final answer, only when action is 'final_answer'")


class Plan(BaseModel):
    steps: list[str] = Field(..., description="Ordered sub-tasks. One step if the task is simple.")


class Critique(BaseModel):
    score: int = Field(..., ge=1, le=10, description="Overall quality from 1 (useless) to 10 (perfect)")
    verdict: Literal["accept", "revise"]
    issues: list[str] = Field(default_factory=list, description="Concrete problems to fix")


class StepRecord(BaseModel):
    thought: str
    action: str
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    ok: bool = True
    duration_ms: float = 0.0


class AgentResult(BaseModel):
    task: str
    answer: str
    steps: list[StepRecord] = Field(default_factory=list)
    success: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tools_used(self) -> list[str]:
        return [s.action for s in self.steps if s.action != "final_answer"]


class RunResult(BaseModel):
    task: str
    answer: str
    plan: list[str] = Field(default_factory=list)
    subtasks: list[AgentResult] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)
    memories_used: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    run_id: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tools_used(self) -> list[str]:
        return [t for r in self.subtasks for t in r.tools_used]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_steps(self) -> int:
        return sum(len(r.steps) for r in self.subtasks)
