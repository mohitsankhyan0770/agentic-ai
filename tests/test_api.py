from conftest import ScriptedLLM, final
from fastapi.testclient import TestClient

from agentic_ai.api import create_app
from agentic_ai.orchestrator import Orchestrator


def test_run_endpoint(registry, settings):
    settings = settings.model_copy(update={"enable_planner": False, "enable_critic": False})
    client = TestClient(create_app(Orchestrator(ScriptedLLM([final("pong")]), registry, settings=settings)))
    assert client.get("/health").json() == {"status": "ok"}
    body = client.post("/run", json={"task": "ping"}).json()
    assert body["answer"] == "pong" and body["total_steps"] == 1
    assert {t["name"] for t in client.get("/tools").json()} >= {"calculator", "read_file"}


def test_rejects_empty_task(registry, settings):
    client = TestClient(create_app(Orchestrator(ScriptedLLM([]), registry, settings=settings)))
    assert client.post("/run", json={"task": ""}).status_code == 422
