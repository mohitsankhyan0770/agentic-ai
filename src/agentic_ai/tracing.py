"""Structured run traces: every plan, tool call, critique and revision is an event.

Events stream to an optional callback (the CLI uses this for live output) and
are written as JSONL under `runs/` so any run can be inspected after the fact.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Event:
    type: str
    data: dict[str, Any]
    t: float = field(default_factory=time.time)


Listener = Callable[[Event], None]


class Tracer:
    def __init__(self, trace_dir: Path | None = None, listener: Listener | None = None) -> None:
        self.run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.events: list[Event] = []
        self.listener = listener
        self._path = None
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._path = trace_dir / f"{self.run_id}.jsonl"

    def emit(self, type_: str, **data: Any) -> None:
        event = Event(type_, data)
        self.events.append(event)
        if self.listener:
            self.listener(event)
        if self._path:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(event), default=str) + "\n")

    @property
    def path(self) -> Path | None:
        return self._path
