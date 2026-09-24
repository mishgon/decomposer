"""Metrics for a finished Workplace Assistant run (`gyms/workplace_assistant/run.py`)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.common import binary_pass_metrics, read_json, read_jsonl
from gyms.workplace_assistant.run import validate_result

COMPLETION_MARKER = ".eval_done.json"


def compute(run_dir: Path) -> dict[str, Any]:
    """Pass metrics for a complete run, next to Gym's own aggregate metrics.

    The rollout grid is re-validated with the gym's check, so the metrics cover
    exactly the split, limit and repeats the run promised.
    """
    marker_path = run_dir / COMPLETION_MARKER
    if not marker_path.is_file():
        raise FileNotFoundError(f"{run_dir} has no {COMPLETION_MARKER}; the run is not complete")
    marker = read_json(marker_path)
    if marker.get("state") != "complete":
        raise ValueError(f"{marker_path} is not a completed run (state {marker.get('state')!r})")
    rollouts_path = run_dir / "rollouts.jsonl"
    integrity = validate_result(
        marker["split"], int(marker["num_repeats"]), rollout_path=rollouts_path, limit=marker.get("limit")
    )
    rewards_by_task: dict[int, list[float]] = defaultdict(list)
    for row in read_jsonl(rollouts_path):
        rewards_by_task[row["_ng_task_index"]].append(float(row["reward"]))
    return {
        "gym": "workplace_assistant",
        "run_dir": str(run_dir),
        "run_name": marker.get("run_name"),
        "experiment": marker["experiment"],
        "purpose": marker.get("purpose"),
        "split": marker["split"],
        "num_repeats": marker["num_repeats"],
        "limit": marker.get("limit"),
        "metrics": binary_pass_metrics(rewards_by_task),
        "gym_aggregate_metrics": read_json(Path(integrity["aggregate_metrics"])),
        "integrity": integrity,
    }
