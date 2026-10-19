from pathlib import Path

from .base import Tool, ToolRegistry, ToolResult
from .calculator import calculator_tool, safe_eval
from .clock import datetime_tool
from .files import Workspace, file_tools
from .web import web_search_tool


def default_registry(workspace_dir: Path, enable_web: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(calculator_tool)
    registry.register(datetime_tool)
    for tool in file_tools(Workspace(workspace_dir)):
        registry.register(tool)
    if enable_web:
        registry.register(web_search_tool)
    return registry


__all__ = ["Tool", "ToolRegistry", "ToolResult", "Workspace", "default_registry", "safe_eval"]
