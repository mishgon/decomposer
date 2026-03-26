from __future__ import annotations

from sympy import sympify


def symbolic_equal(left: str, right: str) -> bool:
    a = sympify(left)
    b = sympify(right)
    equals = a.equals(b)
    if equals is True:
        return True
    return bool((a - b).simplify() == 0)
