"""A calculator that evaluates arithmetic via the AST — never `eval()`.

`eval()` on model output is remote code execution: a prompt-injected
"__import__('os').system('rm -rf ~')" would run. Here only numeric literals,
arithmetic operators and a whitelist of math functions are allowed.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

from pydantic import BaseModel, Field

from .base import Tool

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan", "floor", "ceil", "factorial")
} | {"abs": abs, "round": round, "min": min, "max": max}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

MAX_EXPONENT = 1000
MAX_FACTORIAL = 1000


class CalculatorError(ValueError):
    pass


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"Exponent too large (>{MAX_EXPONENT})")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and args and args[0] > MAX_FACTORIAL:
            raise CalculatorError(f"factorial argument too large (>{MAX_FACTORIAL})")
        return _FUNCS[node.func.id](*args)
    raise CalculatorError(f"Unsupported expression element: {ast.dump(node)[:60]}")


def safe_eval(expression: str) -> float | int:
    expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/").strip()
    # Models often write numbers like 54,999; Python would read that as a tuple.
    expr = re.sub(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?!\d)", lambda m: m.group(0).replace(",", ""), expr)
    if not expr:
        raise CalculatorError("Empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid syntax: {expression!r}") from exc
    try:
        result = _eval(tree)
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero") from exc
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        return int(result)
    return result


class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="Arithmetic expression, e.g. '(17 * 23) + sqrt(16)'")


def _calculate(expression: str) -> str:
    return str(safe_eval(expression))


calculator_tool = Tool(
    name="calculator",
    description="Evaluate an arithmetic expression exactly. Use for ANY math instead of computing in your head.",
    args_model=CalculatorArgs,
    fn=_calculate,
)
