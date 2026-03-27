from __future__ import annotations

import random


INTERVAL_TEMPLATES = [
    ("0", "1"),
    ("-2", "3"),
    ("1", "5"),
    ("-1", "2"),
]

FINITESET_TEMPLATES = [
    "[1, 2, 3]",
    "[x, y]",
    "[0, 2]",
    "[-1, 1, 3]",
]

UNION_TEMPLATES = [
    ("[Interval(0, 1), FiniteSet(2)]"),
    ("[Interval(-1, 0), Interval(1, 2)]"),
    ("[FiniteSet(0), FiniteSet(1, 2)]"),
    ("[Interval(0, 2), FiniteSet(3)]"),
]

INTERSECTION_TEMPLATES = [
    ("[Interval(0, 2), Interval(1, 3)]"),
    ("[Interval(-2, 1), Interval(0, 4)]"),
    ("[FiniteSet(1, 2, 3), FiniteSet(2, 3, 4)]"),
    ("[Interval(0, 5), FiniteSet(2, 6)]"),
]

COMPLEMENT_TEMPLATES = [
    ("Interval(0, 3)", "Interval(1, 2)"),
    ("FiniteSet(1, 2, 3)", "FiniteSet(2)"),
    ("Interval(-1, 2)", "FiniteSet(0)"),
    ("Interval(0, 4)", "Interval(2, 4)"),
]


def _pick(seed: int, templates: list):
    rng = random.Random(seed)
    if seed < len(templates):
        return templates[seed]
    return rng.choice(templates)


def make_interval_seed(seed: int) -> tuple[str, str]:
    return _pick(seed, INTERVAL_TEMPLATES)


def make_finiteset_seed(seed: int) -> str:
    return _pick(seed, FINITESET_TEMPLATES)


def make_union_seed(seed: int) -> str:
    return _pick(seed, UNION_TEMPLATES)


def make_intersection_seed(seed: int) -> str:
    return _pick(seed, INTERSECTION_TEMPLATES)


def make_complement_seed(seed: int) -> tuple[str, str]:
    return _pick(seed, COMPLEMENT_TEMPLATES)
