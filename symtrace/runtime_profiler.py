from __future__ import annotations

import importlib
import itertools
import threading
import time
from contextlib import contextmanager

from .schemas import CallNode, ExceptionInfo
from .serializers import serialize_value


TRACE_SPECS = {
    "simplify": [
        ("sympy.simplify.simplify", "simplify", "sympy.simplify.simplify.simplify"),
        ("sympy.simplify.simplify", "trigsimp", "sympy.simplify.trigsimp.trigsimp"),
        ("sympy.simplify.simplify", "powsimp", "sympy.simplify.powsimp.powsimp"),
        ("sympy.simplify.simplify", "cancel", "sympy.polys.polytools.cancel"),
        ("sympy.simplify.simplify", "factor", "sympy.polys.polytools.factor"),
        ("sympy.simplify.simplify", "expand", "sympy.core.function.expand"),
        ("sympy.simplify.simplify", "together", "sympy.polys.rationaltools.together"),
        ("sympy.simplify.simplify", "factor_terms", "sympy.core.exprtools.factor_terms"),
        ("sympy.simplify.simplify", "hyperexpand", "sympy.simplify.hyperexpand.hyperexpand"),
    ],
    "expand": [
        ("sympy.core.function", "expand", "sympy.core.function.expand"),
    ],
    "trigsimp": [
        ("sympy.simplify.trigsimp", "trigsimp", "sympy.simplify.trigsimp.trigsimp"),
    ],
    "powsimp": [
        ("sympy.simplify.powsimp", "powsimp", "sympy.simplify.powsimp.powsimp"),
    ],
    "nsimplify": [],
    "hyperexpand": [],
    "combsimp": [],
    "gammasimp": [],
    "cancel": [
        ("sympy.polys.polytools", "cancel", "sympy.polys.polytools.cancel"),
    ],
    "apart": [
        ("sympy.polys.partfrac", "apart", "sympy.polys.partfrac.apart"),
    ],
    "together": [
        ("sympy.polys.rationaltools", "together", "sympy.polys.rationaltools.together"),
    ],
    "radsimp": [
        ("sympy.simplify.radsimp", "radsimp", "sympy.simplify.radsimp.radsimp"),
    ],
    "collect": [],
    "expand_trig": [],
    "expand_log": [],
    "expand_mul": [],
    "expand_power_base": [],
    "expand_power_exp": [],
    "logcombine": [],
    "factor": [
        ("sympy.polys.polytools", "factor", "sympy.polys.polytools.factor"),
        ("sympy.polys.polytools", "factor_list", "sympy.polys.polytools.factor_list"),
        ("sympy.polys.polytools", "_generic_factor", "sympy.polys.polytools._generic_factor"),
        ("sympy.polys.polytools", "_symbolic_factor", "sympy.polys.polytools._symbolic_factor"),
        ("sympy.core.exprtools", "factor_terms", "sympy.core.exprtools.factor_terms"),
        ("sympy.polys.rationaltools", "together", "sympy.polys.rationaltools.together"),
    ],
    "factor_list": [],
    "Poly": [],
    "sqf_part": [],
    "sqf_list": [],
    "discriminant": [],
    "gcd": [],
    "gcd_list": [],
    "lcm": [],
    "resultant": [],
    "terms_gcd": [],
    "roots": [],
    "div": [],
    "rem": [],
    "linsolve": [],
    "nonlinsolve": [],
    "nsolve": [],
    "solve_linear": [],
    "solve_undetermined_coeffs": [],
    "solve_linear_system": [],
    "solve_linear_system_LU": [],
    "solve_poly_system": [],
    "linear_eq_to_matrix": [],
    "groebner": [],
    "GroebnerBasis.reduce": [],
    "diff": [],
    "integrate": [],
    "limit": [],
    "series": [],
    "residue": [],
    "summation": [],
    "reduce_inequalities": [],
    "checksol": [],
    "diophantine": [],
    "unrad": [],
    "Interval": [],
    "FiniteSet": [],
    "Union": [],
    "Intersection": [],
    "Complement": [],
    "simplify_logic": [],
    "solve": [
        ("sympy.solvers.solvers", "solve", "sympy.solvers.solvers.solve"),
        ("sympy.solvers.solvers", "_solve", "sympy.solvers.solvers._solve"),
        ("sympy.solvers.solvers", "_solve_system", "sympy.solvers.solvers._solve_system"),
        ("sympy.solvers.solvers", "solve_linear", "sympy.solvers.solvers.solve_linear"),
        ("sympy.polys.polytools", "factor", "sympy.polys.polytools.factor"),
        ("sympy.polys.polytools", "factor_list", "sympy.polys.polytools.factor_list"),
        ("sympy.polys.rationaltools", "together", "sympy.polys.rationaltools.together"),
    ],
    "solveset": [],
}


class RuntimeProfiler:
    def __init__(self, *, entrypoint_name: str, max_calls: int = 5000, max_depth: int = 50) -> None:
        self.entrypoint_name = entrypoint_name
        self.max_calls = max_calls
        self.max_depth = max_depth
        self.calls: dict[str, CallNode] = {}
        self.roots: list[str] = []
        self._stack: list[str] = []
        self._counter = itertools.count(1)
        self.truncated = False

    def _next_call_id(self) -> str:
        return f"c_{next(self._counter):06d}"

    def _serialize_args(self, args: tuple, kwargs: dict) -> dict:
        data = {}
        for idx, value in enumerate(args[:4]):
            data[f"arg{idx}"] = serialize_value(value)
        for key, value in list(kwargs.items())[:8]:
            data[key] = serialize_value(value)
        return data

    def _enter_call(self, func_id: str, args: tuple, kwargs: dict) -> str | None:
        if len(self.calls) >= self.max_calls or len(self._stack) >= self.max_depth:
            self.truncated = True
            return None
        call_id = self._next_call_id()
        parent_call_id = self._stack[-1] if self._stack else None
        node = CallNode(
            call_id=call_id,
            func_id=func_id,
            parent_call_id=parent_call_id,
            depth=len(self._stack),
            inputs=self._serialize_args(args, kwargs),
            start_ns=time.perf_counter_ns(),
            thread_id=threading.get_ident(),
        )
        self.calls[call_id] = node
        if parent_call_id is None:
            self.roots.append(call_id)
        else:
            self.calls[parent_call_id].subcalls.append(call_id)
        self._stack.append(call_id)
        return call_id

    def _leave_call(self, call_id: str | None, result: object = None, exc: Exception | None = None) -> None:
        if call_id is None:
            return
        if self._stack and self._stack[-1] == call_id:
            self._stack.pop()
        node = self.calls[call_id]
        if exc is not None:
            node.exception = ExceptionInfo(type_name=type(exc).__name__, message=str(exc))
        else:
            node.output = serialize_value(result)
        node.end_ns = time.perf_counter_ns()

    @contextmanager
    def _patched_functions(self):
        originals: list[tuple[object, str, object]] = []
        try:
            for module_name, attr_name, func_id in TRACE_SPECS.get(self.entrypoint_name, []):
                module = importlib.import_module(module_name)
                if not hasattr(module, attr_name):
                    continue
                original = getattr(module, attr_name)
                wrapped = self._wrap_callable(original, func_id)
                originals.append((module, attr_name, original))
                setattr(module, attr_name, wrapped)
            yield
        finally:
            for module, attr_name, original in reversed(originals):
                setattr(module, attr_name, original)

    def _wrap_callable(self, func, func_id: str):
        def wrapped(*args, **kwargs):
            call_id = self._enter_call(func_id, args, kwargs)
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                self._leave_call(call_id, exc=exc)
                raise
            self._leave_call(call_id, result=result)
            return result

        wrapped.__name__ = getattr(func, "__name__", "wrapped")
        wrapped.__qualname__ = getattr(func, "__qualname__", wrapped.__name__)
        return wrapped

    def run(self, func, *args, **kwargs) -> tuple[object, dict[str, CallNode], list[str], bool]:
        with self._patched_functions():
            root_func_id = {
                "simplify": "sympy.simplify.simplify.simplify",
                "trigsimp": "sympy.simplify.trigsimp.trigsimp",
                "powsimp": "sympy.simplify.powsimp.powsimp",
                "radsimp": "sympy.simplify.radsimp.radsimp",
                "nsimplify": "sympy.simplify.simplify.nsimplify",
                "hyperexpand": "sympy.simplify.hyperexpand.hyperexpand",
                "combsimp": "sympy.simplify.combsimp.combsimp",
                "gammasimp": "sympy.simplify.gammasimp.gammasimp",
                "collect": "sympy.simplify.radsimp.collect",
                "expand_mul": "sympy.core.function.expand_mul",
                "expand_trig": "sympy.simplify.trigsimp.expand_trig",
                "expand_log": "sympy.core.function.expand_log",
                "expand_power_base": "sympy.core.function.expand_power_base",
                "expand_power_exp": "sympy.core.function.expand_power_exp",
                "logcombine": "sympy.simplify.simplify.logcombine",
                "cancel": "sympy.polys.polytools.cancel",
                "apart": "sympy.polys.partfrac.apart",
                "together": "sympy.polys.rationaltools.together",
                "expand": "sympy.core.function.expand",
                "factor": "sympy.polys.polytools.factor",
                "factor_list": "sympy.polys.polytools.factor_list",
                "Poly": "sympy.polys.polytools.Poly",
                "sqf_part": "sympy.polys.polytools.sqf_part",
                "sqf_list": "sympy.polys.polytools.sqf_list",
                "discriminant": "sympy.polys.polytools.discriminant",
                "gcd": "sympy.polys.polytools.gcd",
                "gcd_list": "sympy.polys.polytools.gcd_list",
                "lcm": "sympy.polys.polytools.lcm",
                "resultant": "sympy.polys.polytools.resultant",
                "terms_gcd": "sympy.polys.polytools.terms_gcd",
                "roots": "sympy.polys.polyroots.roots",
                "div": "sympy.polys.polytools.div",
                "rem": "sympy.polys.polytools.rem",
                "linsolve": "sympy.solvers.solveset.linsolve",
                "nonlinsolve": "sympy.solvers.solveset.nonlinsolve",
                "nsolve": "sympy.solvers.solvers.nsolve",
                "solve_linear": "sympy.solvers.solvers.solve_linear",
                "solve_undetermined_coeffs": "sympy.solvers.solvers.solve_undetermined_coeffs",
                "solve_linear_system": "sympy.solvers.solvers.solve_linear_system",
                "solve_linear_system_LU": "sympy.solvers.solvers.solve_linear_system_LU",
                "solve_poly_system": "sympy.solvers.polysys.solve_poly_system",
                "linear_eq_to_matrix": "sympy.solvers.solveset.linear_eq_to_matrix",
                "groebner": "sympy.polys.polytools.groebner",
                "GroebnerBasis.reduce": "sympy.polys.polytools.GroebnerBasis.reduce",
                "diff": "sympy.core.function.diff",
                "integrate": "sympy.integrals.integrals.integrate",
                "limit": "sympy.series.limits.limit",
                "series": "sympy.series.series.series",
                "residue": "sympy.series.residues.residue",
                "summation": "sympy.concrete.summations.summation",
                "solve": "sympy.solvers.solvers.solve",
                "solveset": "sympy.solvers.solveset.solveset",
                "reduce_inequalities": "sympy.solvers.inequalities.reduce_inequalities",
                "checksol": "sympy.solvers.solvers.checksol",
                "diophantine": "sympy.solvers.diophantine.diophantine.diophantine",
                "unrad": "sympy.solvers.solvers.unrad",
                "Interval": "sympy.sets.sets.Interval",
                "FiniteSet": "sympy.sets.sets.FiniteSet",
                "Union": "sympy.sets.sets.Union",
                "Intersection": "sympy.sets.sets.Intersection",
                "Complement": "sympy.sets.sets.Complement",
                "simplify_logic": "sympy.logic.boolalg.simplify_logic",
            }[self.entrypoint_name]
            call_id = self._enter_call(root_func_id, args, kwargs)
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                self._leave_call(call_id, exc=exc)
                raise
            self._leave_call(call_id, result=result)
        return result, self.calls, self.roots, self.truncated
