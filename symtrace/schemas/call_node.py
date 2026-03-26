from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ExceptionInfo:
    type_name: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class CallNode:
    call_id: str
    func_id: str
    parent_call_id: str | None
    depth: int
    inputs: dict
    output: dict | None = None
    exception: ExceptionInfo | None = None
    subcalls: list[str] = field(default_factory=list)
    start_ns: int = 0
    end_ns: int = 0
    thread_id: int = 0

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_ns"] = self.duration_ns
        return data
