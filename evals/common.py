"""Metric and file helpers shared by the per-gym evaluations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

METRICS_FILENAME = "eval_metrics.json"


def binary_pass_metrics(rewards_by_task: Mapping[Hashable, Sequence[float]]) -> dict[str, Any]:
    """Pass metrics over tasks that each ran `k` times with a reward in {0, 1}.

    `pass_at_1` is the share of passing rollouts, `pass_at_k` the share of tasks
    passed at least once, and `pass_pow_k` the share of tasks passed on every
    repeat. `k` is the repeat count when every task has the same number.
    """
    rollouts = [reward for rewards in rewards_by_task.values() for reward in rewards]
    repeats = sorted({len(rewards) for rewards in rewards_by_task.values()})
    tasks = len(rewards_by_task)
    passed_tasks = [
        [reward == 1.0 for reward in rewards] for rewards in rewards_by_task.values()
    ]
    return {
        "tasks": tasks,
        "repeats": repeats[0] if len(repeats) == 1 else repeats,
        "rollouts": len(rollouts),
        "passed_rollouts": sum(1 for reward in rollouts if reward == 1.0),
        "mean_reward": sum(rollouts) / len(rollouts) if rollouts else 0.0,
        "pass_at_1": (
            sum(1 for reward in rollouts if reward == 1.0) / len(rollouts) if rollouts else 0.0
        ),
        "pass_at_k": sum(1 for passes in passed_tasks if any(passes)) / tasks if tasks else 0.0,
        "pass_pow_k": sum(1 for passes in passed_tasks if all(passes)) / tasks if tasks else 0.0,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    """Write `value` atomically, the way the gym runners write their markers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
