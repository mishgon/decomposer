from __future__ import annotations

import random


DIFF_TEMPLATES = [
    "x**3 + 2*x",
    "sin(x) * cos(x)",
    "exp(x) + x**2",
    "(x**2 + 1)/(x + 1)",
    "log(x) + x**4",
]

INTEGRATE_TEMPLATES = [
    "2*x + 3",
    "cos(x)",
    "x**2 + x",
    "exp(x)",
    "1/(x + 1)",
]

LIMIT_TEMPLATES = [
    "sin(x)/x",
    "(cos(x) - 1)/x",
    "(x**2 - 1)/(x - 1)",
    "(exp(x) - 1)/x",
    "(1 - cos(x))/x**2",
]

SERIES_TEMPLATES = [
    "sin(x)",
    "cos(x)",
    "exp(x)",
    "log(1 + x)",
    "1/(1 - x)",
]


def _pick(seed: int, templates: list[str]) -> str:
    rng = random.Random(seed)
    if seed < len(templates):
        return templates[seed]
    return rng.choice(templates)


def make_diff_seed(seed: int) -> str:
    return _pick(seed, DIFF_TEMPLATES)


def make_integrate_seed(seed: int) -> str:
    return _pick(seed, INTEGRATE_TEMPLATES)


def make_limit_seed(seed: int) -> str:
    return _pick(seed, LIMIT_TEMPLATES)


def make_series_seed(seed: int) -> str:
    return _pick(seed, SERIES_TEMPLATES)
