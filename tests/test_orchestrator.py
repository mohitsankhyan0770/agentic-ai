import json

from conftest import BagOfWordsEmbedder, ScriptedLLM, final, tool

from agentic_ai.memory import InMemoryMemory
from agentic_ai.orchestrator import Orchestrator

ACCEPT = {"score": 9, "verdict": "accept", "issues": []}


def test_simple_task_skips_decomposition(registry, settings):
    llm = ScriptedLLM([{"steps": ["what is 2+2"]}, tool("calculator", expression="2+2"), final("4"), ACCEPT])
    result = Orchestrator(llm, registry, settings=settings).run("what is 2+2")
    assert result.answer == "4" and len(result.subtasks) == 1 and result.tools_used == ["calculator"]


def test_multi_step_passes_results_forward(registry, settings):
    llm = ScriptedLLM(
        [
            {"steps": ["compute 12*12", "save the result to sq.txt"]},
            tool("calculator", expression="12*12"),
            final("144"),
            tool("write_file", path="sq.txt", content="144"),
            final("saved"),
            "12 × 12 = 144 and it was saved to sq.txt.",  # synthesis
            ACCEPT,
        ]
    )
    result = Orchestrator(llm, registry, settings=settings).run("square 12 and save it")
    assert result.plan == ["compute 12*12", "save the result to sq.txt"]
    # Step 2's prompt must contain step 1's result — the original lost this context.
    step2_prompt = llm.calls[3][1]["content"]
    assert "144" in step2_prompt and "Overall goal" in step2_prompt
    assert (settings.workspace_dir / "sq.txt").read_text().strip() == "144"


def test_critic_revision_loop_is_bounded(registry, settings):
    settings = settings.model_copy(update={"max_revisions": 1, "enable_planner": False})
    revise = {"score": 3, "verdict": "revise", "issues": ["missing units"]}
    llm = ScriptedLLM([final("5"), revise, "5 km", revise])  # still unhappy after 1 revision → stop
    result = Orchestrator(llm, registry, settings=settings).run("distance?")
    assert result.answer == "5 km" and len(result.critiques) == 2


def test_low_score_overrides_accept_verdict(registry, settings):
    settings = settings.model_copy(update={"enable_planner": False, "max_revisions": 1})
    inconsistent = {"score": 2, "verdict": "accept", "issues": ["wrong"]}
    llm = ScriptedLLM([final("bad"), inconsistent, "better", ACCEPT])
    result = Orchestrator(llm, registry, settings=settings).run("q")
    assert result.critiques[0].verdict == "revise" and result.answer == "better"


def test_critic_receives_tool_evidence(registry, settings):
    settings = settings.model_copy(update={"enable_planner": False})
    llm = ScriptedLLM([tool("calculator", expression="17*23"), final("391"), ACCEPT])
    Orchestrator(llm, registry, settings=settings).run("17*23")
    critic_prompt = llm.calls[-1][1]["content"]
    assert "EVIDENCE" in critic_prompt and "-> 391" in critic_prompt


def test_memory_is_stored_and_recalled(registry, settings):
    settings = settings.model_copy(update={"enable_planner": False, "enable_critic": False})
    memory = InMemoryMemory(BagOfWordsEmbedder())
    orch = Orchestrator(ScriptedLLM([final("Your name is Mohit."), final("Mohit")]), registry, memory, settings)
    orch.run("my name is Mohit, remember it")
    second = orch.run("what is my name, remember?")
    assert memory.count() == 2 and second.memories_used and "Mohit" in second.memories_used[0]


def test_trace_written(registry, settings):
    settings = settings.model_copy(update={"enable_planner": False, "enable_critic": False})
    result = Orchestrator(ScriptedLLM([final("ok")]), registry, settings=settings).run("hi")
    lines = (settings.trace_dir / f"{result.run_id}.jsonl").read_text().splitlines()
    types = [json.loads(line)["type"] for line in lines]
    assert types[0] == "start" and types[-1] == "done"


def test_planner_failure_falls_back_to_direct(registry, settings):
    settings = settings.model_copy(update={"enable_critic": False})
    llm = ScriptedLLM(["not json at all", final("direct")])
    result = Orchestrator(llm, registry, settings=settings).run("task")
    assert result.plan == ["task"] and result.answer == "direct"
