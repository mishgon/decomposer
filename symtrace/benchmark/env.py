from __future__ import annotations

from sympy import Symbol, sympify

from ..serializers import serialize_value
from ..sympy_wrappers import call_tool


class HandleEnv:
    def __init__(self, initial_expr: str) -> None:
        self.vars = {"$0": sympify(initial_expr)}
        self.next_handle = 1

    def call(self, tool: str, **kwargs) -> dict:
        resolved = {}
        for key, value in kwargs.items():
            if isinstance(value, str) and value.startswith("$"):
                resolved[key] = str(self.vars[value])
            else:
                resolved[key] = value
        result = call_tool(tool, **resolved)
        handle = f"${self.next_handle}"
        self.next_handle += 1
        self.vars[handle] = result
        return {"result": handle, "value": serialize_value(result)}
