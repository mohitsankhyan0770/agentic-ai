"""Agentic AI: a local, planning, tool-using, self-critiquing LLM agent."""

from .agent import Agent
from .config import Settings
from .orchestrator import Orchestrator, build_orchestrator
from .schemas import AgentResult, RunResult

__version__ = "1.0.0"
__all__ = ["Agent", "AgentResult", "Orchestrator", "RunResult", "Settings", "build_orchestrator"]
