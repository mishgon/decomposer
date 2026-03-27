from __future__ import annotations

import random


SOLVE_TEMPLATES = [
    ("x**2 - 1", ["x"]),
    ("[x + y - 2, x - y]", ["x", "y"]),
    ("x/(x - 1) - 2", ["x"]),
    ("sin(x) - 1", ["x"]),
    ("[x + y - 3, 2*x - y]", ["x", "y"]),
    ("(x**2 - 4)/(x - 2) - 4", ["x"]),
]

SOLVESET_TEMPLATES = [
    ("x**2 - 1", ["x"]),
    ("x/(x - 1) - 2", ["x"]),
    ("sin(x) - 1", ["x"]),
    ("(x**2 - 4)/(x - 2) - 4", ["x"]),
    ("x**2 + 1", ["x"]),
    ("exp(x) - 3", ["x"]),
]


def make_equation_seed(seed: int) -> tuple[str, list[str]]:
    rng = random.Random(seed)
    if seed < len(SOLVE_TEMPLATES):
        return SOLVE_TEMPLATES[seed]

    if rng.random() < 0.7:
        return rng.choice(SOLVE_TEMPLATES)

    a = rng.randint(1, 4)
    b = rng.randint(-4, 4)
    c = rng.randint(-5, 5)
    expr = f"{a}*x**2 + {b}*x + {c}"
    return expr, ["x"]


def make_solveset_seed(seed: int) -> tuple[str, list[str]]:
    rng = random.Random(seed)
    if seed < len(SOLVESET_TEMPLATES):
        return SOLVESET_TEMPLATES[seed]

    if rng.random() < 0.8:
        return rng.choice(SOLVESET_TEMPLATES)

    a = rng.randint(1, 3)
    b = rng.randint(-3, 3)
    c = rng.randint(-4, 4)
    expr = f"{a}*x**2 + {b}*x + {c}"
    return expr, ["x"]
