"""Unit tests for the arithmetic calculator tool.

The verifier layer in week 2 reuses `evaluate()` to re-execute LLM
arithmetic claims, so these tests intentionally cover both happy paths
and hostile inputs.
"""
from __future__ import annotations

import pytest

from banna_agent.tools.calculator import evaluate, make_calculator_tool


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1 + 2", 3.0),
        ("10 - 4", 6.0),
        ("6 * 7", 42.0),
        ("10 / 4", 2.5),
        ("10 // 4", 2.0),
        ("10 % 3", 1.0),
        ("2 ** 10", 1024.0),
        ("-5", -5.0),
        ("+5", 5.0),
        ("(1 + 2) * 3", 9.0),
        ("sqrt(16)", 4.0),
        ("abs(-7)", 7.0),
        ("round(3.6)", 4.0),
        ("min(3, 1, 2)", 1.0),
        ("max(3, 1, 2)", 3.0),
        ("log(exp(1))", 1.0),
        ("pi", pytest.approx(3.141592653589793)),
        ("e", pytest.approx(2.718281828459045)),
    ],
)
def test_evaluate_valid_expressions(expr: str, expected: float) -> None:
    assert evaluate(expr) == expected


# ---------------------------------------------------------------------------
# Adversarial inputs — the calculator must refuse to execute arbitrary code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",     # function call to non-whitelist
        "[1, 2, 3]",                          # list literal
        "{'a': 1}",                           # dict literal
        "x + 1",                              # free variable
        "print('hi')",                        # disallowed builtin
        "open('f')",                          # filesystem
        "lambda x: x",                        # lambda
        "1 if True else 0",                   # ternary
        "1 == 1",                             # comparison
        "1 and 2",                            # bool op
    ],
)
def test_evaluate_rejects_hostile_expressions(expr: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        evaluate(expr)


# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


def test_calculator_tool_handler_round_trip() -> None:
    tool = make_calculator_tool()
    assert tool.name == "calculator"
    result = tool.handler({"expression": "2 * (3 + 4)"})
    assert result == {"expression": "2 * (3 + 4)", "value": 14.0}


def test_calculator_tool_rejects_empty_expression() -> None:
    tool = make_calculator_tool()
    with pytest.raises(ValueError, match="expression"):
        tool.handler({"expression": ""})


def test_calculator_schema_shape() -> None:
    tool = make_calculator_tool()
    assert tool.input_schema["required"] == ["expression"]
    assert tool.input_schema["additionalProperties"] is False


def test_calculator_capabilities_pure() -> None:
    tool = make_calculator_tool()
    assert tool.capabilities == frozenset({"pure"})
