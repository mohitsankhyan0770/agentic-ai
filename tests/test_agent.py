from conftest import ScriptedLLM, final, tool

from agentic_ai.agent import Agent
from agentic_ai.llm import LLMError


def test_direct_answer(registry):
    agent = Agent(ScriptedLLM([final("Hello!")]), registry)
    result = agent.run("say hi")
    assert result.success and result.answer == "Hello!" and result.tools_used == []


def test_chains_multiple_tools(registry):
    llm = ScriptedLLM(
        [
            tool("calculator", expression="17 * 23"),
            tool("write_file", path="result.txt", content="391"),
            tool("read_file", path="result.txt"),
            final("17 × 23 = 391, saved to result.txt"),
        ]
    )
    result = Agent(llm, registry).run("compute 17*23 and save it")
    assert result.tools_used == ["calculator", "write_file", "read_file"]
    assert result.steps[0].observation == "391"
    assert result.steps[2].observation.strip() == "391"
    # The model must have been shown the observation before its next decision.
    assert any("Observation: 391" in m["content"] for m in llm.calls[1])


def test_recovers_from_tool_error(registry):
    llm = ScriptedLLM([tool("calculator", expression="2 +* 2"), tool("calculator", expression="2 + 2"), final("4")])
    result = Agent(llm, registry).run("2+2")
    assert not result.steps[0].ok and result.steps[1].ok and result.answer == "4"


def test_duplicate_call_detected(registry):
    llm = ScriptedLLM([tool("calculator", expression="1+1"), tool("calculator", expression="1+1"), final("2")])
    result = Agent(llm, registry).run("1+1")
    assert "already made this exact call" in result.steps[1].observation


def test_step_budget_forces_final_answer(registry):
    llm = ScriptedLLM([tool("calculator", expression=f"{i}+1") for i in range(3)] + ["best effort answer"])
    result = Agent(llm, registry, max_steps=3).run("loop forever")
    assert not result.success and result.answer == "best effort answer" and len(result.steps) == 3


def test_llm_failure_is_reported(registry):
    result = Agent(ScriptedLLM([LLMError("server down")]), registry).run("anything")
    assert not result.success and "server down" in result.answer
