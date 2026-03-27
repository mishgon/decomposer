from __future__ import annotations

import random


INEQUALITY_TEMPLATES = [
    "x**2 - 1 > 0",
    "x - 3 <= 0",
    "x**2 - 4 <= 0",
    "x + 2 > 0",
]

CHECKSOL_TEMPLATES = [
    ("x**2 - 1", "x", "1"),
    ("x**2 - 4", "x", "2"),
    ("x + 3", "x", "-3"),
    ("x**2 + x - 2", "x", "1"),
]

DIOPHANTINE_TEMPLATES = [
    "x + y - 3",
    "2*x + y - 5",
    "x - 2*y - 1",
    "x + 2*y - 4",
]


def _pick(seed: int, templates: list):
    rng = random.Random(seed)
    if seed < len(templates):
        return templates[seed]
    return rng.choice(templates)


def make_inequality_seed(seed: int) -> str:
    return _pick(seed, INEQUALITY_TEMPLATES)


def make_checksol_seed(seed: int) -> tuple[str, str, str]:
    return _pick(seed, CHECKSOL_TEMPLATES)


def make_diophantine_seed(seed: int) -> str:
    return _pick(seed, DIOPHANTINE_TEMPLATES)
