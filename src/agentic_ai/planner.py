"""Planner: decomposes a goal into ordered sub-tasks — but only when that helps.

The original planned *every* input (even "hi"), multiplying latency and making
each step lose the user's actual goal. Here simple tasks come back as a single
step and run directly, and the plan is capped.
"""

from __future__ import annotations

from .llm import LLM, LLMError
from .schemas import Plan

PLANNER_PROMPT = """You are a planner for a tool-using agent.
The agent's tools: {tools}.

Split the user's task into the MINIMUM number of ordered, self-contained sub-tasks
(at most {max_steps}). Each sub-task must be actionable by the agent on its own.

If the task can be done directly in one go (a question, a single calculation,
a greeting), return exactly one step: the task itself, unchanged.

Reply as JSON: {{"steps": ["...", "..."]}}"""


class Planner:
    def __init__(self, llm: LLM, tool_names: list[str], max_steps: int = 5) -> None:
        self.llm = llm
        self.tool_names = tool_names
        self.max_steps = max_steps

    def plan(self, task: str, context: str = "") -> Plan:
        prompt = PLANNER_PROMPT.format(tools=", ".join(self.tool_names), max_steps=self.max_steps)
        user = task if not context else f"Relevant memory:\n{context}\n\nTask: {task}"
        try:
            plan = self.llm.complete_json(
                [{"role": "system", "content": prompt}, {"role": "user", "content": user}], Plan
            )
        except LLMError:
            return Plan(steps=[task])  # planning is an optimisation; never block on it
        steps = [s.strip() for s in plan.steps if s and s.strip()][: self.max_steps]
        return Plan(steps=steps or [task])
