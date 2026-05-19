"""Deterministic arithmetic calculator — no LLM involved, no network.

Restricted-eval instead of sympy: keeps the dep list small and is fast.
The expression is parsed with `ast`, and only a small whitelist of node
types is allowed. This matters because the *calculator verifier* we ship
in week 2 will reuse this module to re-execute LLM-produced arithmetic
claims — so it must be bulletproof against code injection.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from .base import JsonTool

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil, "pi": math.pi, "e": math.e,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_FUNCS:
        return _ALLOWED_FUNCS[node.id]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError(f"disallowed call: {ast.dump(node)}")
        fn = _ALLOWED_FUNCS[node.func.id]
        args = [_evaluate(a) for a in node.args]
        return fn(*args)
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def evaluate(expr: str) -> float:
    """Evaluate an arithmetic expression. Raises `ValueError` on anything
    outside the whitelisted grammar. Safe for untrusted input."""
    tree = ast.parse(expr, mode="eval")
    return float(_evaluate(tree))


def _handler(args: dict[str, Any]) -> dict[str, Any]:
    expr = args.get("expression", "")
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("'expression' must be a non-empty string")
    value = evaluate(expr)
    return {"expression": expr, "value": value}


CALCULATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": (
                "Arithmetic expression to evaluate. Supports +, -, *, /, //, %, ** "
                "and functions: abs, round, min, max, sqrt, log, log2, log10, exp, "
                "sin, cos, tan, floor, ceil, plus constants pi, e. No variables."
            ),
        }
    },
    "required": ["expression"],
    "additionalProperties": False,
}


def make_calculator_tool() -> JsonTool:
    return JsonTool(
        name="calculator",
        description="Evaluate an arithmetic expression deterministically.",
        input_schema=CALCULATOR_SCHEMA,
        handler=_handler,
        capabilities=frozenset({"pure"}),
    )
