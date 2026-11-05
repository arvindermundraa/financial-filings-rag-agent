import pytest

from finrag.calculator import CalculatorError, evaluate


@pytest.mark.parametrize("expr,expected", [
    ("1577 / 32765 * 100", 1577 / 32765 * 100),
    ("(7,207 - 6,000) / 6,000", 1207 / 6000),
    ("$1,577", 1577.0),
    ("-3 + 2 ** 3", 5.0),
    ("10 % 4", 2.0),
])
def test_arithmetic(expr, expected):
    assert evaluate(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')",
    "open('x')",
    "a + 1",
    "[1, 2]",
    "2 ** 1000",
    "1 / 0",
    "True + 1",
    "",
])
def test_rejects_anything_but_arithmetic(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)
