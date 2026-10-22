import pytest

from agentic_ai.tools.calculator import CalculatorError, safe_eval


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 2", 4),
        ("17 * 23", 391),
        ("(10 - 4) / 3", 2),
        ("2 ^ 10", 1024),
        ("sqrt(144) + abs(-3)", 15),
        ("-5 + +2", -3),
        ("10 // 3", 3),
        ("round(pi, 2)", 3.14),
    ],
)
def test_valid_expressions(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "lambda: 1",
        "x + 1",
        "'a' * 10",
    ],
)
def test_rejects_code_injection(payload):
    with pytest.raises(CalculatorError):
        safe_eval(payload)


def test_rejects_resource_exhaustion():
    with pytest.raises(CalculatorError):
        safe_eval("9 ** 9 ** 9")
    with pytest.raises(CalculatorError):
        safe_eval("factorial(100000)")


def test_division_by_zero():
    with pytest.raises(CalculatorError, match="zero"):
        safe_eval("1 / 0")


def test_thousand_separators():
    assert safe_eval("54,999 * 0.18") == pytest.approx(9899.82)
