"""Build a checksum-pinned Workplace teacher comparison."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gyms.workplace_assistant.experiments import (
    SPLIT_ROWS,
    get_experiment,
    output_dir,
    preparation_manifest,
)

BASELINE_EXPERIMENT = "deepseek-v4-flash-0731-qwen35-4b-non-thinking"
CANDIDATE_EXPERIMENT = "qwen36-35b-a3b-teacher-qwen35-4b-non-thinking"
COMPARISON_SPLIT = "validation"
COMPARISON_NUM_REPEATS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], *, source: Path
) -> dict[tuple[int, int], float]:
    task_count = SPLIT_ROWS[COMPARISON_SPLIT]
    expected = task_count * COMPARISON_NUM_REPEATS
    if len(rows) != expected:
        raise ValueError(f"{source}: expected {expected} rows, found {len(rows)}")
    rewards: dict[tuple[int, int], float] = {}
    counts: Counter[int] = Counter()
    for line_number, row in enumerate(rows, 1):
        task = row.get("_ng_task_index")
        rollout = row.get("_ng_rollout_index")
        reward = row.get("reward")
        if isinstance(task, bool) or not isinstance(task, int) or not 0 <= task < task_count:
            raise ValueError(f"{source}:{line_number}: invalid task index {task!r}")
        if (
            isinstance(rollout, bool)
            or not isinstance(rollout, int)
            or not 0 <= rollout < COMPARISON_NUM_REPEATS
        ):
            raise ValueError(
                f"{source}:{line_number}: invalid rollout index {rollout!r}"
            )
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError(f"{source}:{line_number}: reward must be binary")
        normalized_reward = float(reward)
        if normalized_reward not in (0.0, 1.0):
            raise ValueError(f"{source}:{line_number}: reward must be binary")
        key = (task, rollout)
        if key in rewards:
            raise ValueError(f"{source}:{line_number}: duplicate rollout {key}")
        rewards[key] = normalized_reward
        counts[task] += 1
    if counts != Counter({task: COMPARISON_NUM_REPEATS for task in range(task_count)}):
        raise ValueError(f"{source}: task coverage is incomplete")
    return rewards


def _dataset_sha256(experiment_name: str) -> str:
    path = preparation_manifest(COMPARISON_SPLIT, experiment_name)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    value = manifest.get("datasets", {}).get("decomposer", {}).get("sha256")
    if not isinstance(value, str):
        raise ValueError(f"Prepared dataset checksum is missing: {path}")
    return value


def _summary(
    experiment_name: str,
    directory: Path,
    *,
    require_completion: bool,
) -> tuple[dict[str, Any], dict[tuple[int, int], float]]:
    marker = directory / ".eval_done.json"
    if require_completion and not marker.is_file():
        raise FileNotFoundError(f"Completion marker is missing: {marker}")
    rollouts = directory / "rollouts.jsonl"
    rewards = _validate_rows(_read_jsonl(rollouts), source=rollouts)
    passed_rollouts = int(sum(rewards.values()))
    passed_tasks = sum(
        any(rewards[(task, rollout)] == 1.0 for rollout in range(3))
        for task in range(SPLIT_ROWS[COMPARISON_SPLIT])
    )
    source: dict[str, Any] = {
        "rollouts": str(rollouts),
        "rollouts_sha256": _sha256(rollouts),
        "prepared_dataset_sha256": _dataset_sha256(experiment_name),
    }
    if marker.is_file():
        source.update(
            {
                "completion_marker": str(marker),
                "completion_marker_sha256": _sha256(marker),
            }
        )
    return (
        {
            "experiment": experiment_name,
            "source": source,
            "metrics": {
                "scenario_count": SPLIT_ROWS[COMPARISON_SPLIT],
                "rollout_rows": len(rewards),
                "passed_rollouts": passed_rollouts,
                "rollout_success_rate": passed_rollouts / len(rewards),
                "passed_tasks": passed_tasks,
                "task_pass_at_3": passed_tasks / SPLIT_ROWS[COMPARISON_SPLIT],
            },
        },
        rewards,
    )


def build_teacher_comparison(candidate_directory: Path) -> dict[str, Any]:
    baseline_experiment = get_experiment(BASELINE_EXPERIMENT)
    baseline_directory = output_dir(
        baseline_experiment,
        COMPARISON_SPLIT,
        COMPARISON_NUM_REPEATS,
        purpose="trace-generation",
    )
    baseline, baseline_rewards = _summary(
        BASELINE_EXPERIMENT, baseline_directory, require_completion=True
    )
    candidate, candidate_rewards = _summary(
        CANDIDATE_EXPERIMENT, candidate_directory, require_completion=False
    )
    baseline_dataset = baseline["source"]["prepared_dataset_sha256"]
    candidate_dataset = candidate["source"]["prepared_dataset_sha256"]
    if baseline_dataset != candidate_dataset:
        raise ValueError("Workplace teacher comparison datasets do not match")

    outcomes = Counter()
    for key in sorted(baseline_rewards):
        deepseek = baseline_rewards[key] == 1.0
        qwen = candidate_rewards[key] == 1.0
        outcomes[
            "both_correct"
            if deepseek and qwen
            else "deepseek_only"
            if deepseek
            else "qwen_only"
            if qwen
            else "both_incorrect"
        ] += 1
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    return {
        "schema_version": 1,
        "split": COMPARISON_SPLIT,
        "scenario_count": SPLIT_ROWS[COMPARISON_SPLIT],
        "num_repeats": COMPARISON_NUM_REPEATS,
        "prepared_dataset_sha256": baseline_dataset,
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": {
            "rollout_success_rate": (
                candidate_metrics["rollout_success_rate"]
                - baseline_metrics["rollout_success_rate"]
            ),
            "task_pass_at_3": (
                candidate_metrics["task_pass_at_3"]
                - baseline_metrics["task_pass_at_3"]
            ),
        },
        "paired_rollout_outcomes": dict(sorted(outcomes.items())),
    }
