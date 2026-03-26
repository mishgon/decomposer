from __future__ import annotations

from sympy import Symbol, cancel, expand, factor, powsimp, simplify, solve, sympify, trigsimp
from sympy.core.exprtools import factor_terms
from sympy.polys.polytools import factor_list
from sympy.polys.rationaltools import together
from sympy.solvers.solvers import solve_linear


def _coerce_expr(expr: object) -> object:
    if isinstance(expr, str):
        return sympify(expr)
    return expr


def sympy_expand(expr: object) -> object:
    return expand(_coerce_expr(expr))


def sympy_factor(expr: object) -> object:
    return factor(_coerce_expr(expr))


def sympy_simplify(expr: object) -> object:
    return simplify(_coerce_expr(expr))


def sympy_trigsimp(expr: object) -> object:
    return trigsimp(_coerce_expr(expr))


def sympy_powsimp(expr: object) -> object:
    return powsimp(_coerce_expr(expr))


def sympy_cancel(expr: object) -> object:
    return cancel(_coerce_expr(expr))


def sympy_together(expr: object, deep: bool = False) -> object:
    return together(_coerce_expr(expr), deep=deep)


def sympy_factor_terms(expr: object) -> object:
    return factor_terms(_coerce_expr(expr))


def sympy_rational_simplify(expr: object) -> object:
    value = _coerce_expr(expr)
    return factor_terms(cancel(together(powsimp(value), deep=True)))


def sympy_trig_simplify(expr: object) -> object:
    return trigsimp(_coerce_expr(expr), deep=True)


def sympy_solve(expr: object, symbol_names: list[str]) -> object:
    symbols = [Symbol(name) for name in symbol_names]
    return solve(_coerce_expr(expr), *symbols)


def sympy_solve_linear(expr: object, symbol_name: str = "x") -> object:
    symbol = Symbol(symbol_name)
    return solve_linear(_coerce_expr(expr), symbol)


def sympy_factor_list(expr: object) -> object:
    return factor_list(_coerce_expr(expr))


def sympy_factor_decompose(expr: object) -> object:
    return factor(_coerce_expr(expr))


def sympy_solve_branch(expr: object, symbol_names: list[str]) -> object:
    return sympy_solve_branch_with_branch(expr, symbol_names=symbol_names, branch=None)


def sympy_inspect_branches(expr: object, symbol_names: list[str]) -> list[dict[str, str]]:
    expr_obj = _coerce_expr(expr)
    symbols = [Symbol(name) for name in symbol_names]
    return _build_branch_records(expr_obj, symbols)


def sympy_solve_branch_with_branch(
    expr: object,
    symbol_names: list[str],
    branch: str | None = None,
    branch_id: str | None = None,
) -> object:
    expr_obj = _coerce_expr(expr)
    symbols = [Symbol(name) for name in symbol_names]
    branches = _build_branch_records(expr_obj, symbols)
    if branch_id is not None:
        for item in branches:
            if item["branch_id"] == branch_id:
                return item
        raise ValueError(f"Unknown branch_id: {branch_id}")
    if branch is not None:
        for item in branches:
            if item["binding"] == branch:
                return item
        return {"branch_id": "custom", "binding": branch, "resolved_value": branch}
    return branches[0] if branches else {}


def _build_branch_records(expr_obj: object, symbols: list[Symbol]) -> list[dict[str, str]]:
    solved = solve(expr_obj, *symbols, dict=True)
    branches: list[dict[str, str]] = []
    for idx, item in enumerate(solved):
        if isinstance(item, dict):
            binding = "{" + ", ".join(f"{key}: {value}" for key, value in item.items()) + "}"
        elif len(symbols) == 1:
            binding = "{" + f"{symbols[0]}: {item}" + "}"
        else:
            binding = str(item)
        branches.append(
            {
                "branch_id": f"b{idx}",
                "binding": binding,
                "resolved_value": binding,
            }
        )
    return branches


ENTRYPOINTS = {
    "simplify": lambda expr: simplify(expr),
    "factor": lambda expr: factor(expr),
    "expand": lambda expr: expand(expr),
    "solve": lambda expr, symbol_names: solve(expr, *[Symbol(name) for name in symbol_names]),
}

TOOL_FUNCTIONS = {
    "sympy.expand": sympy_expand,
    "sympy.factor": sympy_factor,
    "sympy.simplify": sympy_simplify,
    "sympy.trigsimp": sympy_trigsimp,
    "sympy.powsimp": sympy_powsimp,
    "sympy.cancel": sympy_cancel,
    "sympy.together": sympy_together,
    "sympy.factor_terms": sympy_factor_terms,
    "sympy.rational_simplify": sympy_rational_simplify,
    "sympy.trig_simplify": sympy_trig_simplify,
    "sympy.factor_list": sympy_factor_list,
    "sympy.factor_decompose": sympy_factor_decompose,
    "sympy.solve": sympy_solve,
    "sympy.inspect_branches": sympy_inspect_branches,
    "sympy.solve_linear": sympy_solve_linear,
    "sympy.solve_branch": sympy_solve_branch_with_branch,
}


def call_tool(tool_name: str, **kwargs) -> object:
    try:
        tool = TOOL_FUNCTIONS[tool_name]
    except KeyError as exc:
        raise ValueError(f"Unknown tool: {tool_name}") from exc
    return tool(**kwargs)
