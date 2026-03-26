from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class FunctionDef:
    func_id: str
    qualname: str
    module: str
    class_name: str | None
    file: str
    start_line: int
    end_line: int
    signature: str
    source: str
    ast_hash: str
    docstring_summary: str | None = None
    visibility: str = "private"
    kind: str = "function"

    def to_dict(self) -> dict:
        return asdict(self)
