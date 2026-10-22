"""Orchestrator: wires memory → planner → executor → synthesis → critic → memory.

Key fix vs. the original: each sub-task now receives the overall goal *and the
results of earlier steps*. Before, step 3 had no idea what steps 1–2 produced,
so multi-step tasks couldn't actually build on each other.
"""

from __future__ import annotations

import time

from .agent import Agent
from .config import Settings
from .critic import Critic
from .llm import LLM, LLMError, OllamaLLM
from .memory import ChromaMemory, Memory, OllamaEmbedder
from .planner import Planner
from .schemas import AgentResult, RunResult
from .tools import ToolRegistry, default_registry
from .tracing import Listener, Tracer

SYNTH_PROMPT = """Combine the sub-task results into one clear, complete answer to the user's
original task. Be faithful to the results (especially numbers and tool outputs);
do not invent anything new. Return only the answer."""


def _evidence(results: list[AgentResult], limit: int = 4000) -> str:
    lines = [
        f"[{s.action}] {s.action_input} -> {s.observation}"
        for r in results
        for s in r.steps
        if s.observation is not None
    ]
    text = "\n".join(lines)
    return text[-limit:]


class Orchestrator:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        memory: Memory | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.settings = settings or Settings()

    def run(self, task: str, listener: Listener | None = None) -> RunResult:
        s = self.settings
        tracer = Tracer(s.trace_dir, listener)
        start = time.perf_counter()
        tracer.emit("start", task=task, model=getattr(self.llm, "model", "?"))

        # 1. Recall
        memories: list[str] = []
        if self.memory and s.enable_memory:
            try:
                memories = [h.text for h in self.memory.search(task, k=3)]
            except Exception as exc:  # memory is best-effort
                tracer.emit("error", where="memory.search", message=str(exc))
            if memories:
                tracer.emit("memory_recall", memories=memories)

        memory_ctx = ""
        if memories:
            lines = "\n".join(f"- {m}" for m in memories)
            memory_ctx = f"Past sessions (background only; do NOT redo or continue these tasks):\n{lines}"

        # 2. Plan
        steps = [task]
        if s.enable_planner:
            steps = Planner(self.llm, self.tools.names(), s.max_plan_steps).plan(task).steps
            tracer.emit("plan", steps=steps)

        # 3. Execute
        agent = Agent(self.llm, self.tools, s.max_steps, tracer)
        results: list[AgentResult] = []
        if len(steps) == 1:
            results.append(agent.run(steps[0] if steps[0] else task, context=memory_ctx))
            answer = results[0].answer
        else:
            for i, step in enumerate(steps, 1):
                done = "\n".join(f"Step {j} ({r.task}) result: {r.answer}" for j, r in enumerate(results, 1))
                ctx = "\n\n".join(
                    part
                    for part in (
                        f"Overall goal: {task}",
                        f"Relevant memory:\n{memory_ctx}" if memory_ctx else "",
                        f"Completed so far:\n{done}" if done else "",
                    )
                    if part
                )
                tracer.emit("subtask_start", index=i, total=len(steps), step=step)
                results.append(agent.run(step, context=ctx))
            answer = self._synthesize(task, results, tracer)

        # 4. Critique / revise (bounded)
        critiques = []
        if s.enable_critic:
            critic = Critic(self.llm, s.accept_score)
            evidence = _evidence(results)
            for attempt in range(s.max_revisions + 1):
                critique = critic.review(task, answer, evidence)
                critiques.append(critique)
                tracer.emit("critique", attempt=attempt, **critique.model_dump())
                if critique.verdict == "accept" or attempt == s.max_revisions:
                    break
                try:
                    answer = critic.revise(task, answer, evidence, critique)
                    tracer.emit("revision", answer=answer)
                except LLMError as exc:
                    tracer.emit("error", where="critic.revise", message=str(exc))
                    break

        # 5. Remember
        if self.memory and s.enable_memory:
            try:
                self.memory.add(f"Task: {task}\nAnswer: {answer[:1000]}")
            except Exception as exc:
                tracer.emit("error", where="memory.add", message=str(exc))

        duration = time.perf_counter() - start
        tracer.emit("done", answer=answer, duration_s=round(duration, 2))
        return RunResult(
            task=task,
            answer=answer,
            plan=steps,
            subtasks=results,
            critiques=critiques,
            memories_used=memories,
            duration_s=duration,
            run_id=tracer.run_id,
        )

    def _synthesize(self, task: str, results: list[AgentResult], tracer: Tracer) -> str:
        body = "\n\n".join(f"Sub-task: {r.task}\nResult: {r.answer}" for r in results)
        try:
            answer = self.llm.complete(
                [
                    {"role": "system", "content": SYNTH_PROMPT},
                    {"role": "user", "content": f"Original task: {task}\n\n{body}"},
                ]
            )
        except LLMError as exc:
            tracer.emit("error", where="synthesize", message=str(exc))
            answer = body
        tracer.emit("synthesis", answer=answer)
        return answer


def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Production wiring: Ollama LLM, default tools, persistent Chroma memory."""
    s = settings or Settings.from_env()
    llm = OllamaLLM(s.model, s.ollama_host, s.temperature)
    tools = default_registry(s.workspace_dir)
    memory: Memory | None = None
    if s.enable_memory:
        try:
            memory = ChromaMemory(s.memory_dir, OllamaEmbedder(s.embed_model, s.ollama_host))
        except ImportError:
            memory = None
    return Orchestrator(llm, tools, memory, s)
