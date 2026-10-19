from conftest import ScriptedLLM, final, tool

from agentic_ai import cli
from agentic_ai.orchestrator import Orchestrator


def test_run_once_renders_trace(registry, settings, capsys):
    llm = ScriptedLLM(
        [
            {"steps": ["calc", "save"]},
            tool("calculator", thought="need math", expression="6*7"),
            final("42"),
            final("noted"),
            "The answer is 42.",
            {"score": 8, "verdict": "accept", "issues": []},
        ]
    )
    code = cli.run_once(Orchestrator(llm, registry, settings=settings), "6*7 then note it")
    out = capsys.readouterr().out
    assert code == 0
    for expected in ("Plan", "calculator", "need math", "critic: 8/10", "The answer is 42."):
        assert expected in out
