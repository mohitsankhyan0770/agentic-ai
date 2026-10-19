import importlib.util
from pathlib import Path

from agentic_ai.schemas import AgentResult, RunResult, StepRecord

spec = importlib.util.spec_from_file_location("run_evals", Path(__file__).parents[1] / "evals" / "run_evals.py")
run_evals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_evals)


def _result(answer, tools=()):
    steps = [StepRecord(thought="", action=t, observation="x") for t in tools]
    return RunResult(task="t", answer=answer, subtasks=[AgentResult(task="t", answer=answer, steps=steps)])


def test_pass(tmp_path):
    case = {"checks": {"contains": ["64898"], "tools": ["calculator"]}}
    assert run_evals.score(case, _result("Total: ₹64,898", ["calculator"]), tmp_path) == []


def test_fail_reasons(tmp_path):
    case = {"checks": {"contains": ["391"], "tools": ["calculator"], "file": {"path": "a.txt", "contains": "x"}}}
    failures = run_evals.score(case, _result("390"), tmp_path)
    assert len(failures) == 3


def test_tasks_file_is_valid():
    import json

    cases = json.loads((Path(run_evals.HERE) / "tasks.json").read_text())
    assert len({c["id"] for c in cases}) == len(cases)
