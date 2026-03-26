from __future__ import annotations

import random


TERMINALS = ["x", "y", "2", "3"]
UNARY_FUNCTIONS = ["sin", "cos"]
LOOP_TEMPLATES = [
    "(((x + x) + x) + x)",
    "sin((((y + 1) + 1) + 1))",
    "cos(sin(sin((x + 2))))",
    "((sin(x)**2 + cos(x)**2) + (sin(x)**2 + cos(x)**2))",
    "((((x * y) * y) * y) * y)",
    "sin((x + (x + (x + y))))",
]
MULTISTEP_TEMPLATES = [
    "cos(((y + 2) + 3))",
    "(sin((y * 2)) + 3)",
    "(sin((y * 2)))**2",
    "(cos((x + y)))**2",
    "sin((cos(3) * y))",
    "sin(((y * 3) + y))",
    "(sin(y) * x)",
    "(sin(x) * 3)",
    "(sin(y) * y)",
    "cos(sin((x + 2)))",
    "(x + x**2)/(x*sin(y)**2 + x*cos(y)**2)",
    "((sin(x)**2 + cos(x)**2) * (x + 1))/x",
    "((x**2 - 1)/(x - 1))*(sin(y)**2 + cos(y)**2)",
    "sin((x + x + y))",
    "cos(((x * y) * y))",
]
SIMPLE_TEMPLATES = [
    "(x + 1)*(x + 2)",
    "x**2 + 3*x + 2",
    "sin(y) + cos(y)",
    "(x + y) + 2",
]


def make_expr_seed(seed: int) -> str:
    rng = random.Random(seed)
    seeded_templates = LOOP_TEMPLATES + MULTISTEP_TEMPLATES
    if seed < len(seeded_templates):
        return seeded_templates[seed]

    template_roll = rng.random()
    if template_roll < 0.35:
        return rng.choice(LOOP_TEMPLATES)
    if template_roll < 0.8:
        return rng.choice(MULTISTEP_TEMPLATES)
    if template_roll < 0.9:
        return rng.choice(SIMPLE_TEMPLATES)

    expr = rng.choice(TERMINALS)
    for _ in range(rng.randint(2, 4)):
        choice = rng.randint(0, 3)
        if choice == 0:
            expr = f"({expr} + {rng.choice(TERMINALS)})"
        elif choice == 1:
            expr = f"({expr} * {rng.choice(TERMINALS)})"
        elif choice == 2:
            expr = f"({expr})**2"
        else:
            expr = f"{rng.choice(UNARY_FUNCTIONS)}({expr})"
    return expr
