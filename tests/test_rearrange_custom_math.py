"""Custom Math Operation expressions (rearrange_data.evaluate_column_expression).

Update Date: 23 October 2026
Version: 0.4.11
"""

import math

import numpy as np
import pytest

from vaila.rearrange_data import evaluate_column_expression


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("x * 2.5", 10.0),
        ("x / 1000", 0.004),
        ("np.sqrt(x)", 2.0),
        ("x ** 2", 16.0),
        ("np.deg2rad(x)", math.radians(4.0)),
        ("abs(-x) + round(0.4)", 4.0),
    ],
)
def test_documented_examples_use_x(expression: str, expected: float) -> None:
    assert evaluate_column_expression(expression, 4.0) == pytest.approx(expected)


def test_scipy_is_available_when_named() -> None:
    out = evaluate_column_expression("scipy.signal.medfilt(np.array([x, 9.0, x]), 3)", 1.0)
    assert list(out) == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "expression",
    ["open('/etc/passwd').read()", "__import__('os').getcwd()", "exec('y = 1')", "eval('1')"],
)
def test_files_imports_and_exec_are_blocked(expression: str) -> None:
    with pytest.raises(NameError):
        evaluate_column_expression(expression, 1.0)


def test_nan_input_propagates() -> None:
    assert np.isnan(evaluate_column_expression("x * 2", float("nan")))
