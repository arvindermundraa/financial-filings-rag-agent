"""A safe arithmetic tool for the agent.

FinanceBench questions often need a ratio or a percent change (e.g. capex / revenue). LLMs
are unreliable at multi-digit arithmetic, so the agent calls this tool instead. It walks the
Python AST and only allows numbers and + - * / ** % and parentheses: no names, no calls.
"""
from __future__ import annotations

import ast
import operator

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_EXPONENT = 10


class CalculatorError(ValueError):
    pass


def evaluate(expression: str) -> float:
    expression = expression.replace(",", "").replace("$", "").strip()
    if len(expression) > 200:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"invalid expression: {expression!r}") from e
    return _eval(tree.body)


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError("exponent too large")
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise CalculatorError("division by zero")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise CalculatorError(f"unsupported element: {type(node).__name__}")
