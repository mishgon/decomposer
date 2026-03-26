from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class Episode:
    episode_id: str
    entry_func_id: str
    task_family: str
    input_seed: int
    input: dict
    final_output: dict | None
    root_call_id: str | None
    call_ids: list[str]
    status: str
    duration_ms: float
    max_depth: int
    num_calls: int
    pattern_label: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
