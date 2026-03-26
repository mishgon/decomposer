from __future__ import annotations

import json
from pathlib import Path

from ..abstraction import abstract_episode
from ..schemas import TrainingExample


def export_training_example(raw_episode: dict, output_path: str | Path) -> dict:
    episode = raw_episode["episode"]
    abstract_steps = abstract_episode(raw_episode)
    example = TrainingExample(
        task={
            "entry_function": episode["entry_func_id"],
            "input": episode["input"],
            "goal": "match SymPy reference output",
        },
        tools=sorted({step["tool"] for step in abstract_steps}),
        gold_trajectory=_to_dialogue(abstract_steps),
        final_answer=episode["final_output"],
        metadata={
            "episode_id": episode["episode_id"],
            "pattern_label": episode["pattern_label"],
            "status": episode["status"],
        },
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")
    return example.to_dict()


def _to_dialogue(steps: list[dict]) -> list[dict]:
    trajectory: list[dict] = []
    for step in steps:
        trajectory.append({"assistant": {"tool_call": {"tool": step["tool"], "args": step["args"]}}})
        trajectory.append({"tool": {"result": step["result_handle"], "value": step["value"]["display"]}})
    return trajectory
