from __future__ import annotations

from dataclasses import dataclass

from sympy import (
    Complement,
    FiniteSet,
    Interval,
    Intersection,
    Matrix,
    Poly,
    S,
    Symbol,
    Union,
    apart,
    cancel,
    checksol,
    collect,
    combsimp,
    diff,
    discriminant,
    div,
    expand,
    expand_log,
    expand_mul,
    expand_power_base,
    expand_power_exp,
    expand_trig,
    factor,
    factor_list,
    gammasimp,
    gcd,
    gcd_list,
    groebner,
    hyperexpand,
    integrate,
    lcm,
    limit,
    linsolve,
    logcombine,
    nonlinsolve,
    nsimplify,
    nsolve,
    powsimp,
    radsimp,
    rem,
    residue,
    roots,
    series,
    simplify,
    simplify_logic,
    solve,
    solve_poly_system,
    solveset,
    sqf_list,
    sqf_part,
    summation,
    sympify,
    terms_gcd,
    together,
    trigsimp,
)
from sympy.core.exprtools import factor_terms
from sympy.polys.polytools import resultant
from sympy.solvers.diophantine.diophantine import diophantine
from sympy.solvers.inequalities import reduce_inequalities
from sympy.solvers.solvers import (
    solve_linear,
    solve_linear_system,
    solve_linear_system_LU,
    solve_undetermined_coeffs,
    unrad,
)
from sympy.solvers.solveset import linear_eq_to_matrix


@dataclass(frozen=True, slots=True)
class ToolSpec:
    tool_id: str
    target: object
    family: str
    priority: str
    kind: str = "function"
    owner: str | None = None
    callability: str = "trainable"
    semantic_score: int = 6
    returns_structured_object: bool = False
    likely_trace_depth: str = "medium"


def _coerce_expr(expr: object) -> object:
    if isinstance(expr, str):
        return sympify(expr)
    return expr


def _coerce_symbol(name: str | Symbol) -> Symbol:
    return name if isinstance(name, Symbol) else Symbol(str(name))


def _coerce_symbol_names(symbol_names: list[str] | tuple[str, ...] | str | None, default: str = "x") -> list[Symbol]:
    if symbol_names is None:
        return [Symbol(default)]
    if isinstance(symbol_names, str):
        parsed = sympify(symbol_names)
        if isinstance(parsed, (list, tuple)):
            return [_coerce_symbol(item) for item in parsed]
        return [_coerce_symbol(symbol_names)]
    return [_coerce_symbol(item) for item in symbol_names]


def _coerce_system(value: object) -> list[object]:
    parsed = _coerce_expr(value)
    if isinstance(parsed, (list, tuple)):
        return [_coerce_expr(item) for item in parsed]
    return [_coerce_expr(parsed)]


def _coerce_matrix(value: object) -> Matrix:
    parsed = _coerce_expr(value)
    return parsed if isinstance(parsed, Matrix) else Matrix(parsed)


def _coerce_point(value: object) -> object:
    return _coerce_expr(value)


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


def sympy_radsimp(expr: object) -> object:
    return radsimp(_coerce_expr(expr))


def sympy_nsimplify(expr: object) -> object:
    return nsimplify(_coerce_expr(expr))


def sympy_collect(expr: object, symbol_name: str = "x") -> object:
    return collect(_coerce_expr(expr), _coerce_symbol(symbol_name))


def sympy_cancel(expr: object) -> object:
    return cancel(_coerce_expr(expr))


def sympy_apart(expr: object, symbol_name: str = "x") -> object:
    return apart(_coerce_expr(expr), _coerce_symbol(symbol_name))


def sympy_together(expr: object, deep: bool = False) -> object:
    return together(_coerce_expr(expr), deep=deep)


def sympy_expand_mul(expr: object) -> object:
    return expand_mul(_coerce_expr(expr))


def sympy_expand_log(expr: object) -> object:
    return expand_log(_coerce_expr(expr))


def sympy_expand_trig(expr: object) -> object:
    return expand_trig(_coerce_expr(expr))


def sympy_expand_power_base(expr: object) -> object:
    return expand_power_base(_coerce_expr(expr))


def sympy_expand_power_exp(expr: object) -> object:
    return expand_power_exp(_coerce_expr(expr))


def sympy_logcombine(expr: object) -> object:
    return logcombine(_coerce_expr(expr))


def sympy_hyperexpand(expr: object) -> object:
    return hyperexpand(_coerce_expr(expr))


def sympy_combsimp(expr: object) -> object:
    return combsimp(_coerce_expr(expr))


def sympy_gammasimp(expr: object) -> object:
    return gammasimp(_coerce_expr(expr))


def sympy_factor_terms(expr: object) -> object:
    return factor_terms(_coerce_expr(expr))


def sympy_rational_simplify(expr: object) -> object:
    value = _coerce_expr(expr)
    return factor_terms(cancel(together(powsimp(value), deep=True)))


def sympy_trig_simplify(expr: object) -> object:
    return trigsimp(_coerce_expr(expr), deep=True)


def sympy_solve(expr: object, symbol_names: list[str] | str | None = None) -> object:
    return solve(_coerce_expr(expr), *_coerce_symbol_names(symbol_names))


def sympy_solveset(expr: object, symbol_name: str = "x", domain: object | None = None) -> object:
    solved_domain = S.Complexes if domain is None else _coerce_expr(domain)
    return solveset(_coerce_expr(expr), _coerce_symbol(symbol_name), domain=solved_domain)


def sympy_linsolve(system: object, symbol_names: list[str] | str | None = None) -> object:
    return linsolve(_coerce_system(system), *_coerce_symbol_names(symbol_names))


def sympy_nonlinsolve(system: object, symbol_names: list[str] | str | None = None) -> object:
    return nonlinsolve(_coerce_system(system), _coerce_symbol_names(symbol_names))


def sympy_linear_eq_to_matrix(system: object, symbol_names: list[str] | str | None = None) -> object:
    return linear_eq_to_matrix(_coerce_system(system), _coerce_symbol_names(symbol_names))


def sympy_nsolve(expr: object, symbol_name: str = "x", guess: object = 0) -> object:
    return nsolve(_coerce_expr(expr), _coerce_symbol(symbol_name), _coerce_expr(guess))


def sympy_solve_linear(expr: object, symbol_name: str = "x") -> object:
    return solve_linear(_coerce_expr(expr), _coerce_symbol(symbol_name))


def sympy_solve_undetermined_coeffs(
    expr: object,
    coeff_names: list[str] | str,
    symbol_name: str = "x",
) -> object:
    return solve_undetermined_coeffs(
        _coerce_expr(expr),
        _coerce_symbol_names(coeff_names),
        _coerce_symbol(symbol_name),
    )


def sympy_solve_linear_system(matrix_rows: object, symbol_names: list[str] | str | None = None) -> object:
    return solve_linear_system(_coerce_matrix(matrix_rows), *_coerce_symbol_names(symbol_names))


def sympy_solve_linear_system_lu(matrix_rows: object, symbol_names: list[str] | str | None = None) -> object:
    return solve_linear_system_LU(_coerce_matrix(matrix_rows), _coerce_symbol_names(symbol_names))


def sympy_solve_poly_system(system: object, symbol_names: list[str] | str | None = None) -> object:
    return solve_poly_system(_coerce_system(system), *_coerce_symbol_names(symbol_names))


def sympy_reduce_inequalities(inequalities: object, symbol_name: str = "x") -> object:
    return reduce_inequalities(_coerce_system(inequalities), _coerce_symbol(symbol_name))


def sympy_checksol(expr: object, symbol_name: str = "x", candidate: object = 0) -> object:
    return checksol(_coerce_expr(expr), {_coerce_symbol(symbol_name): _coerce_expr(candidate)})


def sympy_unrad(expr: object) -> object:
    return unrad(_coerce_expr(expr))


def sympy_diophantine(expr: object) -> object:
    return diophantine(_coerce_expr(expr))


def sympy_factor_list(expr: object) -> object:
    return factor_list(_coerce_expr(expr))


def sympy_sqf_part(expr: object, symbol_name: str = "x") -> object:
    return sqf_part(_coerce_expr(expr), _coerce_symbol(symbol_name))


def sympy_sqf_list(expr: object, symbol_name: str = "x") -> object:
    return sqf_list(_coerce_expr(expr), _coerce_symbol(symbol_name))


def sympy_poly(expr: object, symbol_names: list[str] | str | None = None) -> object:
    return Poly(_coerce_expr(expr), *_coerce_symbol_names(symbol_names))


def sympy_groebner(system: object, symbol_names: list[str] | str | None = None) -> object:
    return groebner(_coerce_system(system), *_coerce_symbol_names(symbol_names))


def sympy_groebner_reduce(
    basis_exprs: object,
    target_expr: object,
    symbol_names: list[str] | str | None = None,
) -> object:
    basis = groebner(_coerce_system(basis_exprs), *_coerce_symbol_names(symbol_names))
    return basis.reduce(_coerce_expr(target_expr))


def sympy_resultant(expr: object, other: object, symbol_name: str = "x") -> object:
    return resultant(_coerce_expr(expr), _coerce_expr(other), _coerce_symbol(symbol_name))


def sympy_discriminant(expr: object, symbol_name: str = "x") -> object:
    return discriminant(_coerce_expr(expr), _coerce_symbol(symbol_name))


def sympy_gcd(expr: object, other: object) -> object:
    return gcd(_coerce_expr(expr), _coerce_expr(other))


def sympy_gcd_list(exprs: object) -> object:
    return gcd_list(_coerce_system(exprs))


def sympy_lcm(expr: object, other: object) -> object:
    return lcm(_coerce_expr(expr), _coerce_expr(other))


def sympy_terms_gcd(expr: object) -> object:
    return terms_gcd(_coerce_expr(expr))


def sympy_div(expr: object, other: object, symbol_name: str = "x") -> object:
    return div(_coerce_expr(expr), _coerce_expr(other), _coerce_symbol(symbol_name))


def sympy_rem(expr: object, other: object, symbol_name: str = "x") -> object:
    return rem(_coerce_expr(expr), _coerce_expr(other), _coerce_symbol(symbol_name))


def sympy_roots(expr: object, symbol_name: str = "x") -> object:
    return roots(_coerce_expr(expr), _coerce_symbol(symbol_name))


def sympy_diff(expr: object, symbol_name: str = "x", order: int = 1) -> object:
    return diff(_coerce_expr(expr), _coerce_symbol(symbol_name), order)


def sympy_integrate(expr: object, symbol_name: str = "x") -> object:
    return integrate(_coerce_expr(expr), _coerce_symbol(symbol_name))


def sympy_limit(expr: object, symbol_name: str = "x", point: object = 0) -> object:
    return limit(_coerce_expr(expr), _coerce_symbol(symbol_name), _coerce_expr(point))


def sympy_series(expr: object, symbol_name: str = "x", point: object = 0, order: int = 6) -> object:
    return series(_coerce_expr(expr), _coerce_symbol(symbol_name), _coerce_expr(point), order)


def sympy_residue(expr: object, symbol_name: str = "x", point: object = 0) -> object:
    return residue(_coerce_expr(expr), _coerce_symbol(symbol_name), _coerce_expr(point))


def sympy_summation(expr: object, symbol_name: str = "n", lower: object = 0, upper: object = 5) -> object:
    return summation(
        _coerce_expr(expr),
        (_coerce_symbol(symbol_name), _coerce_expr(lower), _coerce_expr(upper)),
    )


def sympy_interval(start: object, end: object, left_open: bool = False, right_open: bool = False) -> object:
    return Interval(_coerce_expr(start), _coerce_expr(end), left_open=left_open, right_open=right_open)


def sympy_finiteset(elements: object) -> object:
    values = _coerce_system(elements)
    return FiniteSet(*values)


def sympy_union(sets: object) -> object:
    return Union(*_coerce_system(sets))


def sympy_intersection(sets: object) -> object:
    return Intersection(*_coerce_system(sets))


def sympy_complement(left: object, right: object) -> object:
    return Complement(_coerce_expr(left), _coerce_expr(right))


def sympy_simplify_logic(expr: object) -> object:
    return simplify_logic(_coerce_expr(expr))


def sympy_inspect_branches(expr: object, symbol_names: list[str] | str | None = None) -> list[dict[str, str]]:
    expr_obj = _coerce_expr(expr)
    symbols = _coerce_symbol_names(symbol_names)
    return _build_branch_records(expr_obj, symbols)


def sympy_solve_branch(expr: object, symbol_names: list[str] | str | None = None) -> object:
    return sympy_solve_branch_with_branch(expr, symbol_names=symbol_names, branch=None)


def sympy_solve_branch_with_branch(
    expr: object,
    symbol_names: list[str] | str | None = None,
    branch: str | None = None,
    branch_id: str | None = None,
) -> object:
    expr_obj = _coerce_expr(expr)
    symbols = _coerce_symbol_names(symbol_names)
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
    "trigsimp": lambda expr: trigsimp(expr),
    "powsimp": lambda expr: powsimp(expr),
    "radsimp": lambda expr: radsimp(expr),
    "nsimplify": lambda expr: nsimplify(expr),
    "hyperexpand": lambda expr: hyperexpand(expr),
    "combsimp": lambda expr: combsimp(expr),
    "gammasimp": lambda expr: gammasimp(expr),
    "collect": lambda expr: collect(expr, Symbol("x")),
    "expand_mul": lambda expr: expand_mul(expr),
    "expand_trig": lambda expr: expand_trig(expr),
    "expand_log": lambda expr: expand_log(expr),
    "expand_power_base": lambda expr: expand_power_base(expr),
    "expand_power_exp": lambda expr: expand_power_exp(expr),
    "logcombine": lambda expr: logcombine(expr),
    "cancel": lambda expr: cancel(expr),
    "apart": lambda expr: apart(expr, Symbol("x")),
    "together": lambda expr: together(expr),
    "factor": lambda expr: factor(expr),
    "factor_list": lambda expr: factor_list(expr),
    "Poly": lambda expr: Poly(expr, Symbol("x")),
    "sqf_part": lambda expr: sqf_part(expr, Symbol("x")),
    "sqf_list": lambda expr: sqf_list(expr, Symbol("x")),
    "discriminant": lambda expr: discriminant(expr, Symbol("x")),
    "gcd": lambda expr, other: gcd(expr, other),
    "gcd_list": lambda exprs: gcd_list(sympify(exprs)),
    "lcm": lambda expr, other: lcm(expr, other),
    "resultant": lambda expr, other: resultant(expr, other, Symbol("x")),
    "terms_gcd": lambda expr: terms_gcd(expr),
    "roots": lambda expr: roots(expr, Symbol("x")),
    "div": lambda expr, other: div(expr, other, Symbol("x")),
    "rem": lambda expr, other: rem(expr, other, Symbol("x")),
    "linsolve": lambda system, symbol_names: linsolve(sympify(system), *[_coerce_symbol(name) for name in symbol_names]),
    "nonlinsolve": lambda system, symbol_names: nonlinsolve(sympify(system), [_coerce_symbol(name) for name in symbol_names]),
    "nsolve": lambda expr, symbol_name, guess: nsolve(sympify(expr), Symbol(symbol_name), sympify(guess)),
    "solve_linear": lambda expr: solve_linear(expr, Symbol("x")),
    "solve_undetermined_coeffs": lambda expr, coeff_names: solve_undetermined_coeffs(
        sympify(expr), [_coerce_symbol(name) for name in sympify(coeff_names)], Symbol("x")
    ),
    "solve_linear_system": lambda matrix_rows, symbol_names: solve_linear_system(
        Matrix(sympify(matrix_rows)), *[_coerce_symbol(name) for name in symbol_names]
    ),
    "solve_linear_system_LU": lambda matrix_rows, symbol_names: solve_linear_system_LU(
        Matrix(sympify(matrix_rows)), [_coerce_symbol(name) for name in symbol_names]
    ),
    "solve_poly_system": lambda system, symbol_names: solve_poly_system(
        sympify(system), *[_coerce_symbol(name) for name in symbol_names]
    ),
    "linear_eq_to_matrix": lambda system, symbol_names: linear_eq_to_matrix(
        sympify(system), [_coerce_symbol(name) for name in symbol_names]
    ),
    "groebner": lambda system, symbol_names: groebner(sympify(system), *[_coerce_symbol(name) for name in symbol_names]),
    "GroebnerBasis.reduce": lambda basis_exprs, target_expr, symbol_names: groebner(
        sympify(basis_exprs), *[_coerce_symbol(name) for name in symbol_names]
    ).reduce(sympify(target_expr)),
    "expand": lambda expr: expand(expr),
    "diff": lambda expr: diff(expr, Symbol("x")),
    "integrate": lambda expr: integrate(expr, Symbol("x")),
    "limit": lambda expr: limit(expr, Symbol("x"), 0),
    "series": lambda expr: series(expr, Symbol("x"), 0, 6),
    "residue": lambda expr, symbol_name, point: residue(sympify(expr), Symbol(symbol_name), sympify(point)),
    "summation": lambda expr, symbol_name, lower, upper: summation(
        sympify(expr), (Symbol(symbol_name), sympify(lower), sympify(upper))
    ),
    "solve": lambda expr, symbol_names: solve(expr, *[_coerce_symbol(name) for name in symbol_names]),
    "solveset": lambda expr, symbol_names: solveset(expr, _coerce_symbol(symbol_names[0]), domain=S.Complexes),
    "reduce_inequalities": lambda inequalities: reduce_inequalities([sympify(inequalities)], Symbol("x")),
    "checksol": lambda expr, symbol_name, candidate: checksol(sympify(expr), {Symbol(symbol_name): sympify(candidate)}),
    "diophantine": lambda expr: diophantine(sympify(expr)),
    "unrad": lambda expr: unrad(sympify(expr)),
    "Interval": lambda start, end: Interval(sympify(start), sympify(end)),
    "FiniteSet": lambda elements: FiniteSet(*sympify(elements)),
    "Union": lambda sets: Union(*sympify(sets)),
    "Intersection": lambda sets: Intersection(*sympify(sets)),
    "Complement": lambda left, right: Complement(sympify(left), sympify(right)),
    "simplify_logic": lambda expr: simplify_logic(sympify(expr)),
}

TOOL_FUNCTIONS = {
    "sympy.simplify": sympy_simplify,
    "sympy.trigsimp": sympy_trigsimp,
    "sympy.powsimp": sympy_powsimp,
    "sympy.radsimp": sympy_radsimp,
    "sympy.nsimplify": sympy_nsimplify,
    "sympy.collect": sympy_collect,
    "sympy.cancel": sympy_cancel,
    "sympy.apart": sympy_apart,
    "sympy.together": sympy_together,
    "sympy.factor": sympy_factor,
    "sympy.expand": sympy_expand,
    "sympy.expand_mul": sympy_expand_mul,
    "sympy.expand_log": sympy_expand_log,
    "sympy.expand_trig": sympy_expand_trig,
    "sympy.expand_power_base": sympy_expand_power_base,
    "sympy.expand_power_exp": sympy_expand_power_exp,
    "sympy.logcombine": sympy_logcombine,
    "sympy.hyperexpand": sympy_hyperexpand,
    "sympy.combsimp": sympy_combsimp,
    "sympy.gammasimp": sympy_gammasimp,
    "sympy.factor_terms": sympy_factor_terms,
    "sympy.rational_simplify": sympy_rational_simplify,
    "sympy.trig_simplify": sympy_trig_simplify,
    "sympy.solve": sympy_solve,
    "sympy.solveset": sympy_solveset,
    "sympy.linsolve": sympy_linsolve,
    "sympy.nonlinsolve": sympy_nonlinsolve,
    "sympy.linear_eq_to_matrix": sympy_linear_eq_to_matrix,
    "sympy.nsolve": sympy_nsolve,
    "sympy.solve_linear": sympy_solve_linear,
    "sympy.solve_undetermined_coeffs": sympy_solve_undetermined_coeffs,
    "sympy.solve_linear_system": sympy_solve_linear_system,
    "sympy.solve_linear_system_LU": sympy_solve_linear_system_lu,
    "sympy.solve_poly_system": sympy_solve_poly_system,
    "sympy.reduce_inequalities": sympy_reduce_inequalities,
    "sympy.checksol": sympy_checksol,
    "sympy.unrad": sympy_unrad,
    "sympy.diophantine": sympy_diophantine,
    "sympy.Poly": sympy_poly,
    "sympy.factor_list": sympy_factor_list,
    "sympy.sqf_part": sympy_sqf_part,
    "sympy.sqf_list": sympy_sqf_list,
    "sympy.groebner": sympy_groebner,
    "sympy.GroebnerBasis.reduce": sympy_groebner_reduce,
    "sympy.resultant": sympy_resultant,
    "sympy.discriminant": sympy_discriminant,
    "sympy.gcd": sympy_gcd,
    "sympy.gcd_list": sympy_gcd_list,
    "sympy.lcm": sympy_lcm,
    "sympy.terms_gcd": sympy_terms_gcd,
    "sympy.div": sympy_div,
    "sympy.rem": sympy_rem,
    "sympy.roots": sympy_roots,
    "sympy.diff": sympy_diff,
    "sympy.integrate": sympy_integrate,
    "sympy.limit": sympy_limit,
    "sympy.series": sympy_series,
    "sympy.residue": sympy_residue,
    "sympy.summation": sympy_summation,
    "sympy.Interval": sympy_interval,
    "sympy.FiniteSet": sympy_finiteset,
    "sympy.Union": sympy_union,
    "sympy.Intersection": sympy_intersection,
    "sympy.Complement": sympy_complement,
    "sympy.simplify_logic": sympy_simplify_logic,
    "sympy.inspect_branches": sympy_inspect_branches,
    "sympy.solve_branch": sympy_solve_branch_with_branch,
}

TOOL_SPECS = [
    ToolSpec("sympy.simplify", simplify, "simplification", "P0", semantic_score=9, likely_trace_depth="high"),
    ToolSpec("sympy.trigsimp", trigsimp, "simplification", "P0", semantic_score=8, likely_trace_depth="high"),
    ToolSpec("sympy.powsimp", powsimp, "simplification", "P0", semantic_score=7),
    ToolSpec("sympy.radsimp", radsimp, "simplification", "P0", semantic_score=7),
    ToolSpec("sympy.nsimplify", nsimplify, "simplification", "P0", semantic_score=7),
    ToolSpec("sympy.collect", collect, "simplification", "P0", semantic_score=8),
    ToolSpec("sympy.cancel", cancel, "simplification", "P0", semantic_score=8),
    ToolSpec("sympy.apart", apart, "simplification", "P0", semantic_score=8),
    ToolSpec("sympy.together", together, "simplification", "P0", semantic_score=7),
    ToolSpec("sympy.factor", factor, "simplification", "P0", semantic_score=9, likely_trace_depth="high"),
    ToolSpec("sympy.expand", expand, "simplification", "P1", semantic_score=8),
    ToolSpec("sympy.expand_mul", expand_mul, "simplification", "P1", semantic_score=6),
    ToolSpec("sympy.expand_log", expand_log, "simplification", "P1", semantic_score=6),
    ToolSpec("sympy.expand_trig", expand_trig, "simplification", "P1", semantic_score=7),
    ToolSpec("sympy.expand_power_base", expand_power_base, "simplification", "P1", semantic_score=6),
    ToolSpec("sympy.expand_power_exp", expand_power_exp, "simplification", "P1", semantic_score=6),
    ToolSpec("sympy.logcombine", logcombine, "simplification", "P1", semantic_score=6),
    ToolSpec("sympy.hyperexpand", hyperexpand, "simplification", "P1", semantic_score=6, likely_trace_depth="high"),
    ToolSpec("sympy.combsimp", combsimp, "simplification", "P2", semantic_score=5),
    ToolSpec("sympy.gammasimp", gammasimp, "simplification", "P2", semantic_score=5),
    ToolSpec("sympy.solve", solve, "solver", "P0", semantic_score=9, returns_structured_object=True, likely_trace_depth="high"),
    ToolSpec("sympy.solveset", solveset, "solver", "P0", semantic_score=8, returns_structured_object=True, likely_trace_depth="high"),
    ToolSpec("sympy.linsolve", linsolve, "solver", "P0", semantic_score=8, returns_structured_object=True),
    ToolSpec("sympy.nonlinsolve", nonlinsolve, "solver", "P0", semantic_score=8, returns_structured_object=True),
    ToolSpec(
        "sympy.linear_eq_to_matrix",
        linear_eq_to_matrix,
        "solver",
        "P0",
        semantic_score=8,
        returns_structured_object=True,
    ),
    ToolSpec("sympy.nsolve", nsolve, "solver", "P1", semantic_score=6),
    ToolSpec("sympy.solve_linear", solve_linear, "solver", "P1", semantic_score=7, returns_structured_object=True),
    ToolSpec(
        "sympy.solve_undetermined_coeffs",
        solve_undetermined_coeffs,
        "solver",
        "P1",
        semantic_score=7,
        returns_structured_object=True,
    ),
    ToolSpec(
        "sympy.solve_linear_system",
        solve_linear_system,
        "solver",
        "P1",
        semantic_score=7,
        returns_structured_object=True,
    ),
    ToolSpec(
        "sympy.solve_linear_system_LU",
        solve_linear_system_LU,
        "solver",
        "P1",
        semantic_score=6,
        returns_structured_object=True,
    ),
    ToolSpec(
        "sympy.solve_poly_system",
        solve_poly_system,
        "solver",
        "P1",
        semantic_score=7,
        returns_structured_object=True,
    ),
    ToolSpec("sympy.reduce_inequalities", reduce_inequalities, "solver", "P1", semantic_score=7),
    ToolSpec("sympy.checksol", checksol, "solver", "P2", semantic_score=5),
    ToolSpec("sympy.unrad", unrad, "solver", "P2", semantic_score=5),
    ToolSpec("sympy.diophantine", diophantine, "solver", "P2", semantic_score=6, returns_structured_object=True),
    ToolSpec("sympy.Poly", Poly, "polynomial", "P0", semantic_score=8, returns_structured_object=True),
    ToolSpec("sympy.factor_list", factor_list, "polynomial", "P0", semantic_score=8, returns_structured_object=True),
    ToolSpec("sympy.sqf_part", sqf_part, "polynomial", "P0", semantic_score=7),
    ToolSpec("sympy.sqf_list", sqf_list, "polynomial", "P0", semantic_score=7, returns_structured_object=True),
    ToolSpec("sympy.groebner", groebner, "polynomial", "P0", semantic_score=9, returns_structured_object=True, likely_trace_depth="high"),
    ToolSpec(
        "sympy.GroebnerBasis.reduce",
        groebner,
        "polynomial",
        "P0",
        kind="method",
        owner="GroebnerBasis",
        semantic_score=8,
        returns_structured_object=True,
    ),
    ToolSpec("sympy.resultant", resultant, "polynomial", "P1", semantic_score=7),
    ToolSpec("sympy.discriminant", discriminant, "polynomial", "P1", semantic_score=6),
    ToolSpec("sympy.gcd", gcd, "polynomial", "P1", semantic_score=6),
    ToolSpec("sympy.gcd_list", gcd_list, "polynomial", "P1", semantic_score=6),
    ToolSpec("sympy.lcm", lcm, "polynomial", "P1", semantic_score=6),
    ToolSpec("sympy.terms_gcd", terms_gcd, "polynomial", "P1", semantic_score=6),
    ToolSpec("sympy.div", div, "polynomial", "P1", semantic_score=6, returns_structured_object=True),
    ToolSpec("sympy.rem", rem, "polynomial", "P1", semantic_score=5),
    ToolSpec("sympy.roots", roots, "polynomial", "P2", semantic_score=6, returns_structured_object=True),
    ToolSpec("sympy.diff", diff, "calculus", "P0", semantic_score=9),
    ToolSpec("sympy.integrate", integrate, "calculus", "P0", semantic_score=9, likely_trace_depth="high"),
    ToolSpec("sympy.limit", limit, "calculus", "P0", semantic_score=8),
    ToolSpec("sympy.series", series, "calculus", "P1", semantic_score=7),
    ToolSpec("sympy.residue", residue, "calculus", "P2", semantic_score=5),
    ToolSpec("sympy.summation", summation, "calculus", "P2", semantic_score=6),
    ToolSpec("sympy.Interval", Interval, "sets", "P1", semantic_score=5, returns_structured_object=True),
    ToolSpec("sympy.FiniteSet", FiniteSet, "sets", "P1", semantic_score=5, returns_structured_object=True),
    ToolSpec("sympy.Union", Union, "sets", "P1", semantic_score=5, returns_structured_object=True),
    ToolSpec("sympy.Intersection", Intersection, "sets", "P1", semantic_score=5, returns_structured_object=True),
    ToolSpec("sympy.Complement", Complement, "sets", "P1", semantic_score=5, returns_structured_object=True),
    ToolSpec("sympy.simplify_logic", simplify_logic, "logic", "P2", semantic_score=5),
]


def call_tool(tool_name: str, **kwargs) -> object:
    try:
        tool = TOOL_FUNCTIONS[tool_name]
    except KeyError as exc:
        raise ValueError(f"Unknown tool: {tool_name}") from exc
    return tool(**kwargs)
