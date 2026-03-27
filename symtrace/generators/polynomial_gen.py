from __future__ import annotations

import random


POLY_FACTOR_TEMPLATES = [
    "(x - 1)**2*(x + 2)",
    "(x + 3)**2*(x - 2)",
    "(x - 2)**3",
    "(x**2 - 1)*(x + 4)",
    "(x - 1)*(x + 1)*(x + 2)",
    "(x + 1)**2*(x + 2)**2",
]

DISCRIMINANT_TEMPLATES = [
    "x**2 + 3*x + 2",
    "x**2 - 4*x + 1",
    "x**3 - 3*x + 1",
    "x**3 + x**2 - x - 1",
]

DIVISION_TEMPLATES = [
    ("x**3 - 1", "x - 1"),
    ("x**4 - 1", "x**2 + 1"),
    ("x**3 + 2*x**2 + x + 2", "x + 2"),
    ("x**4 + x**3 - x - 1", "x + 1"),
]

LINEAR_SYSTEM_TEMPLATES = [
    ("[x + y - 2, x - y]", ["x", "y"]),
    ("[x + 2*y - 5, 3*x - y - 4]", ["x", "y"]),
    ("[2*x + y - 1, x - y - 2]", ["x", "y"]),
    ("[x + y + z - 3, x - y + z - 1, 2*x + z - 4]", ["x", "y", "z"]),
]

GROEBNER_SYSTEM_TEMPLATES = [
    ("[x*y - 1, y - x]", ["x", "y"]),
    ("[x + y - 1, x - y]", ["x", "y"]),
    ("[x**2 - y, y - 1]", ["x", "y"]),
    ("[x*y - 2, x - y]", ["x", "y"]),
]


def make_poly_factor_seed(seed: int) -> str:
    rng = random.Random(seed)
    if seed < len(POLY_FACTOR_TEMPLATES):
        return POLY_FACTOR_TEMPLATES[seed]
    return rng.choice(POLY_FACTOR_TEMPLATES)


def make_discriminant_seed(seed: int) -> str:
    rng = random.Random(seed)
    if seed < len(DISCRIMINANT_TEMPLATES):
        return DISCRIMINANT_TEMPLATES[seed]
    return rng.choice(DISCRIMINANT_TEMPLATES)


def make_division_seed(seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    if seed < len(DIVISION_TEMPLATES):
        return DIVISION_TEMPLATES[seed]
    return rng.choice(DIVISION_TEMPLATES)


def make_linear_system_seed(seed: int) -> tuple[str, list[str]]:
    rng = random.Random(seed)
    if seed < len(LINEAR_SYSTEM_TEMPLATES):
        return LINEAR_SYSTEM_TEMPLATES[seed]
    return rng.choice(LINEAR_SYSTEM_TEMPLATES)


def make_groebner_seed(seed: int) -> tuple[str, list[str]]:
    rng = random.Random(seed)
    if seed < len(GROEBNER_SYSTEM_TEMPLATES):
        return GROEBNER_SYSTEM_TEMPLATES[seed]
    return rng.choice(GROEBNER_SYSTEM_TEMPLATES)
