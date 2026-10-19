"""Command-line interface with a live view of the agent's reasoning."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import Settings
from .llm import LLMError
from .orchestrator import build_orchestrator
from .tracing import Event

console = Console()


def render_event(event: Event) -> None:
    d = event.data
    match event.type:
        case "memory_recall":
            console.print(
                f"[dim]🧠 recalled {len(d['memories'])} memor{'y' if len(d['memories']) == 1 else 'ies'}[/dim]"
            )
        case "plan" if len(d["steps"]) > 1:
            console.print("[bold cyan]📋 Plan[/bold cyan]")
            for i, step in enumerate(d["steps"], 1):
                console.print(f"   {i}. {step}")
        case "subtask_start":
            console.print(f"\n[bold]▶ Step {d['index']}/{d['total']}:[/bold] {d['step']}")
        case "tool_call":
            colour = "green" if d["ok"] else "red"
            console.print(f"   [dim]💭 {d['thought']}[/dim]")
            console.print(f"   [{colour}]🔧 {d['tool']}[/{colour}]({json.dumps(d['input'])}) [dim]{d['ms']}ms[/dim]")
            obs = d["observation"].replace("\n", " ")
            console.print(f"   [dim]↳ {obs[:160]}{'…' if len(obs) > 160 else ''}[/dim]")
        case "critique":
            colour = "green" if d["verdict"] == "accept" else "yellow"
            issues = f" — {'; '.join(d['issues'])}" if d["issues"] else ""
            console.print(f"[{colour}]🔍 critic: {d['score']}/10 {d['verdict']}[/{colour}][dim]{issues}[/dim]")
        case "revision":
            console.print("[yellow]✏️  revised answer[/yellow]")
        case "budget_exhausted":
            console.print("[red]⏱  step budget exhausted[/red]")
        case "error":
            console.print(f"[red]⚠ {d['where']}: {d['message']}[/red]")


def run_once(orchestrator, task: str, quiet: bool = False) -> int:
    try:
        result = orchestrator.run(task, listener=None if quiet else render_event)
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    console.print(Panel(Markdown(result.answer), title="Answer", border_style="cyan"))
    console.print(
        f"[dim]{result.duration_s:.1f}s · {result.total_steps} steps · tools: "
        f"{', '.join(result.tools_used) or 'none'} · run {result.run_id}[/dim]"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-ai", description="Local planning/tool-using/self-critiquing agent")
    parser.add_argument("task", nargs="*", help="Run a single task and exit (omit for interactive chat)")
    parser.add_argument("--model", help="Ollama model (default from AGENT_MODEL)")
    parser.add_argument("--no-planner", action="store_true")
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Only print the final answer")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    overrides = {
        "model": args.model or settings.model,
        "enable_planner": settings.enable_planner and not args.no_planner,
        "enable_critic": settings.enable_critic and not args.no_critic,
        "enable_memory": settings.enable_memory and not args.no_memory,
    }
    settings = settings.model_copy(update=overrides)
    orchestrator = build_orchestrator(settings)

    if args.task:
        return run_once(orchestrator, " ".join(args.task), args.quiet)

    console.print(f"[bold cyan]Agentic AI[/bold cyan] · model [bold]{settings.model}[/bold] · type 'exit' to quit\n")
    while True:
        try:
            task = console.input("[bold]You ›[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if task.lower() in {"exit", "quit"}:
            break
        if task:
            run_once(orchestrator, task, args.quiet)
            console.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
