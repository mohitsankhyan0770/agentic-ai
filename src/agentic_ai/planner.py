"""Planner: decomposes a goal into ordered sub-tasks — but only when that helps.

The original planned *every* input (even "hi"), multiplying latency and making
each step lose the user's actual goal. Here simple tasks come back as a single
step and run directly, and the plan is capped.
"""

from __future__ import annotations

from .llm import LLM, LLMError
from .schemas import Plan

PLANNER_PROMPT = """You are a planner for a tool-using agent.
Available tools (for information only): {tools}.

Break the task into the MINIMUM number of steps (at most {max_steps}).
Each step must be a full instruction in plain English. NEVER output a bare tool name as a step.

Most tasks need only ONE step. If the task is a single question or calculation, return exactly one
step that repeats the task unchanged. Only split when the user asks for several distinct actions.

Examples:
Task: What is 17 * 23?
{{"steps": ["What is 17 * 23?"]}}
Task: Calculate 125 * 8 and save it to result.txt
{{"steps": ["Calculate 125 * 8", "Save the result to result.txt"]}}

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
        names = {n.lower() for n in self.tool_names}
        if not steps or any(s.lower() in names for s in steps):
            return Plan(steps=[task])  # a step that is just a tool name means the plan is junk
        return Plan(steps=steps)