from __future__ import annotations

import random


COLLECT_TEMPLATES = [
    "x*y + x + 2*x**2",
    "x**2 + x*y + x*z",
    "x*sin(y) + x*cos(y) + x",
    "x**3 + x**2*y + x*y",
]

EXPAND_TRIG_TEMPLATES = [
    "sin(x + y)",
    "cos(x + y)",
    "sin(2*x)",
    "cos(2*x)",
]

EXPAND_LOG_TEMPLATES = [
    "log(x*y)",
    "log(x**2)",
    "log((x + 1)*(y + 1))",
    "log((x*y)**2)",
]

LOGCOMBINE_TEMPLATES = [
    "log(x) + log(y)",
    "2*log(x)",
    "log(x) - log(y)",
    "log(x) + log(y) - log(z)",
]


def _pick(seed: int, templates: list[str]) -> str:
    rng = random.Random(seed)
    if seed < len(templates):
        return templates[seed]
    return rng.choice(templates)


def make_collect_seed(seed: int) -> str:
    return _pick(seed, COLLECT_TEMPLATES)


def make_expand_trig_seed(seed: int) -> str:
    return _pick(seed, EXPAND_TRIG_TEMPLATES)


def make_expand_log_seed(seed: int) -> str:
    return _pick(seed, EXPAND_LOG_TEMPLATES)


def make_logcombine_seed(seed: int) -> str:
    return _pick(seed, LOGCOMBINE_TEMPLATES)
