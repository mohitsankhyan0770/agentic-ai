"""Runtime configuration, loaded from environment variables (or a .env file)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so we don't need an extra dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    model: str = Field("llama3.2:3b", description="Ollama chat model")
    embed_model: str = Field("nomic-embed-text", description="Ollama embedding model")
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.0

    max_steps: int = Field(6, ge=1, description="Max tool-use iterations per (sub)task")
    max_plan_steps: int = Field(5, ge=1)
    max_revisions: int = Field(1, ge=0, description="How many times the critic may request a rewrite")
    accept_score: int = Field(7, ge=1, le=10, description="Critic score needed to accept an answer")

    enable_planner: bool = True
    enable_critic: bool = True
    enable_memory: bool = True

    workspace_dir: Path = Path("workspace")
    memory_dir: Path = Path(".agent_memory")
    trace_dir: Path | None = Path("runs")

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()
        trace = os.getenv("AGENT_TRACE_DIR", "runs")
        return cls(
            model=os.getenv("AGENT_MODEL", cls.model_fields["model"].default),
            embed_model=os.getenv("AGENT_EMBED_MODEL", cls.model_fields["embed_model"].default),
            ollama_host=os.getenv("OLLAMA_HOST", cls.model_fields["ollama_host"].default),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0")),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
            max_plan_steps=int(os.getenv("AGENT_MAX_PLAN_STEPS", "5")),
            max_revisions=int(os.getenv("AGENT_MAX_REVISIONS", "1")),
            accept_score=int(os.getenv("AGENT_ACCEPT_SCORE", "7")),
            enable_planner=_env_bool("AGENT_ENABLE_PLANNER", True),
            enable_critic=_env_bool("AGENT_ENABLE_CRITIC", True),
            enable_memory=_env_bool("AGENT_ENABLE_MEMORY", True),
            workspace_dir=Path(os.getenv("AGENT_WORKSPACE", "workspace")),
            memory_dir=Path(os.getenv("AGENT_MEMORY_DIR", ".agent_memory")),
            trace_dir=Path(trace) if trace else None,
        )
