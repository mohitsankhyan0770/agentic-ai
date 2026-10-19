from agentic_ai.tools import ToolRegistry
from agentic_ai.tools.calculator import calculator_tool


def test_unknown_tool_returns_error_not_exception(registry):
    res = registry.execute("teleport", {})
    assert not res.ok and "Unknown tool" in res.output


def test_invalid_args_are_reported(registry):
    res = registry.execute("calculator", {"expr": "1+1"})
    assert not res.ok and "expression" in res.output


def test_tool_exception_is_captured(registry):
    res = registry.execute("calculator", {"expression": "1/0"})
    assert not res.ok and "zero" in res.output.lower()


def test_file_roundtrip(registry):
    assert registry.execute("write_file", {"path": "notes/a.txt", "content": "hello"}).ok
    assert registry.execute("write_file", {"path": "notes/a.txt", "content": "world", "mode": "append"}).ok
    assert registry.execute("read_file", {"path": "notes/a.txt"}).output == "hello\nworld\n"
    assert "notes/a.txt" in registry.execute("list_files", {"path": "notes"}).output


def test_path_traversal_blocked(registry):
    for path in ("../escape.txt", "/etc/passwd", "a/../../b.txt"):
        res = registry.execute("write_file", {"path": path, "content": "x"})
        assert not res.ok and "escapes the workspace" in res.output


def test_duplicate_registration_rejected():
    reg = ToolRegistry()
    reg.register(calculator_tool)
    try:
        reg.register(calculator_tool)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_observation_truncation():
    from agentic_ai.tools.base import ToolResult

    assert ToolResult(True, "x" * 5000).as_observation(limit=100).endswith("[truncated]")
