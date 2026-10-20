"""Evaluation harness: run a fixed task suite against a real model and score it.

Usage:
    python evals/run_evals.py                       # full agent, default model
    python evals/run_evals.py --model qwen2.5:7b
    python evals/run_evals.py --ablate              # compare full vs no-planner vs no-critic
    python evals/run_evals.py --offline             # skip tasks that need the internet
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agentic_ai.config import Settings
from agentic_ai.orchestrator import build_orchestrator
from agentic_ai.schemas import RunResult

HERE = Path(__file__).parent
console = Console()

ABLATIONS = {
    "full": {},
    "no-planner": {"enable_planner": False},
    "no-critic": {"enable_critic": False},
    "bare": {"enable_planner": False, "enable_critic": False},
}


def _norm(text: str) -> str:
    return text.lower().replace(",", "").replace("₹", "")


def score(case: dict, result: RunResult, workspace: Path) -> list[str]:
    """Return a list of failed checks (empty list = pass)."""
    checks, answer, failures = case.get("checks", {}), _norm(result.answer), []
    for s in checks.get("contains", []):
        if _norm(s) not in answer:
            failures.append(f"answer missing '{s}'")
    if "contains_any" in checks and not any(_norm(s) in answer for s in checks["contains_any"]):
        failures.append(f"answer has none of {checks['contains_any']}")
    for s in checks.get("not_contains", []):
        if _norm(s) in answer:
            failures.append(f"answer contains forbidden '{s}'")
    for tool in checks.get("tools", []):
        if tool not in result.tools_used:
            failures.append(f"did not use {tool}")
    for tool in checks.get("forbidden_tools", []):
        if tool in result.tools_used:
            failures.append(f"used forbidden tool {tool}")
    if "file" in checks:
        f = workspace / checks["file"]["path"]
        if not f.exists() or _norm(checks["file"]["contains"]) not in _norm(f.read_text()):
            failures.append(f"file {checks['file']['path']} missing expected content")
    return failures


def run_suite(settings: Settings, cases: list[dict], label: str) -> dict:
    rows = []
    for case in cases:
        ws = Path(tempfile.mkdtemp(prefix="eval-ws-"))
        for name, content in case.get("setup", {}).items():
            (ws / name).write_text(content)
        s = settings.model_copy(update={"workspace_dir": ws, "enable_memory": False, "trace_dir": None})
        orch = build_orchestrator(s)
        t0 = time.perf_counter()
        try:
            result = orch.run(case["task"])
            failures = score(case, result, ws)
            steps, tools = result.total_steps, result.tools_used
        except Exception as exc:
            failures, steps, tools = [f"crashed: {exc}"], 0, []
        rows.append(
            {
                "id": case["id"],
                "passed": not failures,
                "failures": failures,
                "latency_s": round(time.perf_counter() - t0, 2),
                "steps": steps,
                "tools": tools,
            }
        )
        mark = "[green]PASS[/green]" if not failures else f"[red]FAIL[/red] {'; '.join(failures)}"
        console.print(f"  [{label}] {case['id']:<18} {mark}")
        shutil.rmtree(ws, ignore_errors=True)

    return {
        "config": label,
        "model": settings.model,
        "pass_rate": sum(r["passed"] for r in rows) / len(rows),
        "median_latency_s": statistics.median(r["latency_s"] for r in rows),
        "avg_steps": statistics.mean(r["steps"] for r in rows),
        "cases": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--ablate", action="store_true", help="Run every ablation config")
    ap.add_argument("--offline", action="store_true", help="Skip tasks needing internet")
    ap.add_argument("--only", help="Comma-separated task ids")
    args = ap.parse_args()

    settings = Settings.from_env()
    if args.model:
        settings = settings.model_copy(update={"model": args.model})
    cases = json.loads((HERE / "tasks.json").read_text())
    if args.offline:
        cases = [c for c in cases if not c.get("needs_network")]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    configs = ABLATIONS if args.ablate else {"full": {}}
    summaries = []
    for label, overrides in configs.items():
        console.print(f"\n[bold]Config: {label}[/bold] ({settings.model})")
        summaries.append(run_suite(settings.model_copy(update=overrides), cases, label))

    table = Table(title=f"Eval results · {settings.model} · {len(cases)} tasks")
    for col in ("config", "pass rate", "median latency", "avg steps"):
        table.add_column(col)
    for s in summaries:
        table.add_row(s["config"], f"{s['pass_rate']:.0%}", f"{s['median_latency_s']:.1f}s", f"{s['avg_steps']:.1f}")
    console.print(table)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{settings.model.replace(':', '_').replace('/', '_')}-{time.strftime('%Y%m%d-%H%M')}.json"
    out.write_text(json.dumps(summaries, indent=2))
    console.print(f"[dim]Saved {out}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
