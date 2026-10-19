"""File tools confined to a workspace directory (no path traversal)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .base import Tool

MAX_READ_BYTES = 50_000
MAX_WRITE_BYTES = 200_000


class SandboxError(PermissionError):
    pass


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise SandboxError(f"Path '{relative}' escapes the workspace")
        return candidate


class ReadArgs(BaseModel):
    path: str = Field(..., description="File path relative to the workspace")


class WriteArgs(BaseModel):
    path: str = Field(..., description="File path relative to the workspace")
    content: str = Field(..., description="Text to write")
    mode: Literal["overwrite", "append"] = "overwrite"


class ListArgs(BaseModel):
    path: str = Field(".", description="Directory relative to the workspace")


def file_tools(workspace: Workspace) -> list[Tool]:
    def read_file(path: str) -> str:
        target = workspace.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        data = target.read_bytes()[:MAX_READ_BYTES]
        return data.decode("utf-8", errors="replace")

    def write_file(path: str, content: str, mode: str = "overwrite") -> str:
        if len(content.encode()) > MAX_WRITE_BYTES:
            raise ValueError("Content too large")
        target = workspace.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a" if mode == "append" else "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content if content.endswith("\n") else content + "\n")
        return f"Wrote {len(content)} chars to {target.relative_to(workspace.root)}"

    def list_files(path: str = ".") -> str:
        target = workspace.resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        entries = sorted(
            p.relative_to(workspace.root).as_posix() + ("/" if p.is_dir() else "") for p in target.iterdir()
        )
        return "\n".join(entries) or "(empty)"

    return [
        Tool("read_file", "Read a text file from the workspace.", ReadArgs, read_file),
        Tool("write_file", "Write or append text to a file in the workspace.", WriteArgs, write_file),
        Tool("list_files", "List files in a workspace directory.", ListArgs, list_files),
    ]
