from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path

from .schemas import FunctionDef


TARGET_MODULE_PREFIXES = ("sympy.simplify", "sympy.solvers", "sympy.polys", "sympy.core")


def normalize_source(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def _hash_ast(source: str) -> str:
    tree = ast.parse(source)
    normalized = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def function_def_from_callable(func: object) -> FunctionDef:
    module = inspect.getmodule(func)
    if module is None or not module.__name__.startswith(TARGET_MODULE_PREFIXES):
        raise ValueError(f"Unsupported callable for indexing: {func!r}")
    source = normalize_source(inspect.getsource(func))
    lines, start_line = inspect.getsourcelines(func)
    end_line = start_line + len(lines) - 1
    qualname = f"{module.__name__}.{func.__qualname__}"
    return FunctionDef(
        func_id=qualname,
        qualname=qualname,
        module=module.__name__,
        class_name=None,
        file=str(Path(inspect.getsourcefile(func) or "").resolve()),
        start_line=start_line,
        end_line=end_line,
        signature=str(inspect.signature(func)),
        source=source,
        ast_hash=_hash_ast(source),
        docstring_summary=((inspect.getdoc(func) or "").strip().splitlines() or [None])[0],
        visibility="public" if not func.__name__.startswith("_") else "private",
        kind="function",
    )


def build_function_catalog(functions: list[object]) -> list[FunctionDef]:
    return [function_def_from_callable(func) for func in functions]
