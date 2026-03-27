from __future__ import annotations

import random


NSIMPLIFY_TEMPLATES = [
    "0.5",
    "0.3333333333333333",
    "1.4142135623730951",
    "3.141592653589793",
]

RADSIMP_TEMPLATES = [
    "1/sqrt(2)",
    "1/(1 + sqrt(2))",
    "sqrt(8) + sqrt(18)",
    "1/(sqrt(3) + 1)",
]

HYPEREXPAND_TEMPLATES = [
    "hyper((1, 1), (2,), x)",
    "hyper((1/2, 1), (3/2,), x**2)",
]

COMBSIMP_TEMPLATES = [
    "factorial(n + 1)/factorial(n)",
    "binomial(n, k)*factorial(k)",
]

GAMMASIMP_TEMPLATES = [
    "gamma(x + 1)/gamma(x)",
    "gamma(x + 2)/gamma(x + 1)",
]

EXPAND_MUL_TEMPLATES = [
    "(x + 1)*(x + 2)",
    "(x - 1)*(x + y)",
]

EXPAND_POWER_BASE_TEMPLATES = [
    "(x*y)**2",
    "(2*x)**3",
]

EXPAND_POWER_EXP_TEMPLATES = [
    "x**(y + 2)",
    "2**(x + 1)",
]

TERMS_GCD_TEMPLATES = [
    "x**2 + x*y",
    "2*x + 2*y",
]

SIMPLIFY_LOGIC_TEMPLATES = [
    "(A & B) | (A & ~B)",
    "(A | B) & (A | ~B)",
]

NONLINSOLVE_TEMPLATES = [
    ("[x**2 - 1, y - x]", ["x", "y"]),
    ("[x*y - 1, y - x]", ["x", "y"]),
]

NSOLVE_TEMPLATES = [
    ("sin(x)", "x", "3"),
    ("x**2 - 2", "x", "1"),
]

SOLVE_LINEAR_TEMPLATES = [
    "x + 2",
    "2*x - 3",
]

SOLVE_LINEAR_SYSTEM_TEMPLATES = [
    ("[[1, 1, 2], [1, -1, 0]]", ["x", "y"]),
    ("[[2, 1, 5], [3, -1, 4]]", ["x", "y"]),
]

SOLVE_POLY_SYSTEM_TEMPLATES = [
    ("[x + y - 2, x - y]", ["x", "y"]),
    ("[x**2 - y, y - 1]", ["x", "y"]),
]

SOLVE_UNDETERMINED_COEFFS_TEMPLATES = [
    ("a*x + b - (2*x + 3)", ["a", "b"]),
    ("a*x**2 + b*x + c - (x**2 + 2*x + 1)", ["a", "b", "c"]),
]

UNRAD_TEMPLATES = [
    "sqrt(x) - 2",
    "sqrt(x + 1) - 3",
]

RESIDUE_TEMPLATES = [
    ("1/x", "x", "0"),
    ("1/(x - 1)", "x", "1"),
]

SUMMATION_TEMPLATES = [
    ("n", "n", "1", "5"),
    ("n**2", "n", "1", "4"),
]


def _pick(seed: int, templates: list):
    rng = random.Random(seed)
    if seed < len(templates):
        return templates[seed]
    return rng.choice(templates)


def make_nsimplify_seed(seed: int) -> str:
    return _pick(seed, NSIMPLIFY_TEMPLATES)


def make_radsimp_seed(seed: int) -> str:
    return _pick(seed, RADSIMP_TEMPLATES)


def make_hyperexpand_seed(seed: int) -> str:
    return _pick(seed, HYPEREXPAND_TEMPLATES)


def make_combsimp_seed(seed: int) -> str:
    return _pick(seed, COMBSIMP_TEMPLATES)


def make_gammasimp_seed(seed: int) -> str:
    return _pick(seed, GAMMASIMP_TEMPLATES)


def make_expand_mul_seed(seed: int) -> str:
    return _pick(seed, EXPAND_MUL_TEMPLATES)


def make_expand_power_base_seed(seed: int) -> str:
    return _pick(seed, EXPAND_POWER_BASE_TEMPLATES)


def make_expand_power_exp_seed(seed: int) -> str:
    return _pick(seed, EXPAND_POWER_EXP_TEMPLATES)


def make_terms_gcd_seed(seed: int) -> str:
    return _pick(seed, TERMS_GCD_TEMPLATES)


def make_simplify_logic_seed(seed: int) -> str:
    return _pick(seed, SIMPLIFY_LOGIC_TEMPLATES)


def make_nonlinsolve_seed(seed: int) -> tuple[str, list[str]]:
    return _pick(seed, NONLINSOLVE_TEMPLATES)


def make_nsolve_seed(seed: int) -> tuple[str, str, str]:
    return _pick(seed, NSOLVE_TEMPLATES)


def make_solve_linear_seed(seed: int) -> str:
    return _pick(seed, SOLVE_LINEAR_TEMPLATES)


def make_solve_linear_system_seed(seed: int) -> tuple[str, list[str]]:
    return _pick(seed, SOLVE_LINEAR_SYSTEM_TEMPLATES)


def make_solve_poly_system_seed(seed: int) -> tuple[str, list[str]]:
    return _pick(seed, SOLVE_POLY_SYSTEM_TEMPLATES)


def make_solve_undetermined_coeffs_seed(seed: int) -> tuple[str, list[str]]:
    return _pick(seed, SOLVE_UNDETERMINED_COEFFS_TEMPLATES)


def make_unrad_seed(seed: int) -> str:
    return _pick(seed, UNRAD_TEMPLATES)


def make_residue_seed(seed: int) -> tuple[str, str, str]:
    return _pick(seed, RESIDUE_TEMPLATES)


def make_summation_seed(seed: int) -> tuple[str, str, str, str]:
    return _pick(seed, SUMMATION_TEMPLATES)
