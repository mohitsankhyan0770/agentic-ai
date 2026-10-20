"""ReAct executor: Thought → Action → Observation, repeated until a final answer.

The original agent made exactly one tool call and returned the raw result, so
it could never chain tools (search → calculate → save). This loop lets the
model see each observation and decide the next action, with guardrails:
a step budget, duplicate-call detection, and error observations it can recover from.
"""

from __future__ import annotations

import json

from .llm import LLM, LLMError, Message
from .schemas import AgentDecision, AgentResult, StepRecord
from .tools import ToolRegistry
from .tracing import Tracer

FINAL = "final_answer"

SYSTEM_PROMPT = """You are a careful, tool-using assistant. Solve the task step by step.

On every turn reply with ONE JSON object:
{{"thought": "...", "action": "<tool name or final_answer>", "action_input": {{...}}, "answer": null}}

Available tools (name, description, args):
{tools}

Rules:
- Use the calculator for any arithmetic; never do math in your head.
- Only use tools the task actually needs. Never write or read files unless the user asks for it.
- Example: for "What is 6*7?" call calculator, see 42, then reply
  {{"thought": "got it", "action": "final_answer", "action_input": {{}}, "answer": "6*7 = 42"}}
- Use web_search for facts that may be recent or that you are unsure of.
- After each action you will receive an Observation. Use it; do not repeat the same call.
- If a tool returns ERROR, fix the arguments or try another approach.
- When you have enough information, reply with action "final_answer" and put the
  complete answer for the user in "answer". Base it on the observations.
- The context may contain results of earlier steps. Reuse them; never recompute them. 
  If the task is to save something, call write_file with the value from the context.

"""

FORCE_FINAL_PROMPT = (
    "You have used your step budget. Using only the observations so far, give the best "
    "final answer you can now, and say clearly if anything is incomplete."
)


class Agent:
    def __init__(self, llm: LLM, tools: ToolRegistry, max_steps: int = 6, tracer: Tracer | None = None) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.tracer = tracer

    def _emit(self, type_: str, **data) -> None:
        if self.tracer:
            self.tracer.emit(type_, **data)

    def _initial_messages(self, task: str, context: str) -> list[Message]:
        user = f"Task: {task}"
        if context:
            user = f"Context (may be useful):\n{context}\n\n{user}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=self.tools.describe())},
            {"role": "user", "content": user},
        ]

    def run(self, task: str, context: str = "") -> AgentResult:
        messages = self._initial_messages(task, context)
        steps: list[StepRecord] = []
        seen_calls: set[str] = set()

        for _ in range(self.max_steps):
            try:
                decision = self.llm.complete_json(messages, AgentDecision)
            except LLMError as exc:
                self._emit("error", where="agent", message=str(exc))
                return AgentResult(task=task, answer=f"Agent failed: {exc}", steps=steps, success=False)

            messages.append({"role": "assistant", "content": decision.model_dump_json()})

            if decision.action == FINAL:
                answer = (decision.answer or decision.thought).strip()
                steps.append(StepRecord(thought=decision.thought, action=FINAL))
                self._emit("final", task=task, answer=answer)
                return AgentResult(task=task, answer=answer, steps=steps)

            call_key = f"{decision.action}:{json.dumps(decision.action_input, sort_keys=True)}"
            if call_key in seen_calls:
                observation, ok, ms = (
                    "You already made this exact call. Use the earlier observation or give the final answer.",
                    False,
                    0.0,
                )
            else:
                seen_calls.add(call_key)
                result = self.tools.execute(decision.action, decision.action_input)
                observation, ok, ms = result.as_observation(), result.ok, result.duration_ms

            steps.append(
                StepRecord(
                    thought=decision.thought,
                    action=decision.action,
                    action_input=decision.action_input,
                    observation=observation,
                    ok=ok,
                    duration_ms=ms,
                )
            )
            self._emit(
                "tool_call",
                thought=decision.thought,
                tool=decision.action,
                input=decision.action_input,
                observation=observation,
                ok=ok,
                ms=round(ms, 1),
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}\n\n"
                    "Check the task: has EVERY action it asks for been done (for example both "
                    "calculating AND saving to a file)? If yes, reply with action final_answer. "
                    "If something is still missing, call the tool for that missing part.",
                }
            )

        # Budget exhausted: degrade gracefully instead of returning nothing.
        self._emit("budget_exhausted", task=task, steps=len(steps))
        try:
            answer = self.llm.complete(messages + [{"role": "user", "content": FORCE_FINAL_PROMPT}])
        except LLMError as exc:
            answer = f"Agent stopped after {len(steps)} steps without an answer ({exc})."
        return AgentResult(task=task, answer=answer, steps=steps, success=False)
