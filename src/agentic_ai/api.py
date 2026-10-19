"""HTTP API (FastAPI). Run with: uvicorn agentic_ai.api:app --reload"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .llm import LLMError
from .orchestrator import Orchestrator, build_orchestrator
from .schemas import RunResult


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)


def create_app(orchestrator: Orchestrator | None = None) -> FastAPI:
    app = FastAPI(title="Agentic AI", version="1.0.0")
    state: dict[str, Orchestrator] = {}
    if orchestrator is not None:
        state["orch"] = orchestrator

    def get_orch() -> Orchestrator:
        if "orch" not in state:
            state["orch"] = build_orchestrator()
        return state["orch"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tools")
    def tools() -> list[dict]:
        return [t.spec() for t in get_orch().tools.tools.values()]

    @app.post("/run", response_model=RunResult)
    def run(req: RunRequest) -> RunResult:
        try:
            result = get_orch().run(req.task)
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return result

    return app


app = create_app()
