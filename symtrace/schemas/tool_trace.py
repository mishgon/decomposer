from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class AbstractStep:
    tool: str
    args: dict
    result_handle: str
    value: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class TrainingExample:
    task: dict
    tools: list[str]
    gold_trajectory: list[dict]
    final_answer: dict | None
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)
