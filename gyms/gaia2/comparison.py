"""Reuse completed full GAIA2 runs for the pinned held-out comparison."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gyms.gaia2.dataset import sha256_file
from gyms.gaia2.experiments import (
    SCENARIO_COUNT,
    SPLIT_MANIFEST_NAME,
    TEST_SCENARIO_COUNT,
    Experiment,
    get_experiment,
    output_dir,
)
from gyms.gaia2.partition import SPLIT_MANIFEST_SHA256, partition_scenario_ids

COMPARISON_NUM_REPEATS = 3
COMPARISON_BASELINE_EXPERIMENT_NAMES = (
    "deepseek-v4-flash-0731",
    "deepseek-v4-flash-0731-teacher-qwen35-4b-non-thinking",
    "qwen35-2b-base-non-thinking",
    "qwen35-4b-base-non-thinking-qwen35-4b-non-thinking",
    "qwen35-4b-base-non-thinking-teacher-qwen35-4b-non-thinking",
    "qwen35-4b-non-thinking",
    (
        "qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279-"
        "non-thinking-qwen35-4b-non-thinking"
    ),
    (
        "qwen35-4b-sft-mixed-v1-final-493c24c4-404-non-thinking-"
        "qwen35-4b-non-thinking"
    ),
    (
        "qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k-non-thinking-"
        "qwen35-4b-non-thinking"
    ),
    (
        "qwen35-4b-sft-workplace-v1-3765-32k-non-thinking-"
        "qwen35-4b-non-thinking"
    ),
    "qwen35-9b-base-non-thinking",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def _validate_rollouts(
    rows: Sequence[dict[str, Any]],
    *,
    scenario_ids: Sequence[str],
    num_repeats: int,
    source: Path,
) -> None:
    expected_ids = tuple(scenario_ids)
    expected_id_set = set(expected_ids)
    if len(expected_id_set) != len(expected_ids):
        raise ValueError("Expected GAIA2 scenario IDs are not unique")
    expected_rows = len(expected_ids) * num_repeats
    if len(rows) != expected_rows:
        raise ValueError(
            f"{source}: expected {expected_rows} rollout rows, found {len(rows)}"
        )

    counts: Counter[str] = Counter()
    logical_runs: dict[str, set[int]] = {
        scenario_id: set() for scenario_id in expected_ids
    }
    for index, row in enumerate(rows, 1):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in expected_id_set:
            raise ValueError(f"{source}:{index}: unexpected task_id {task_id!r}")
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{source}:{index}: score must be binary")
        if float(score) not in (0.0, 1.0):
            raise ValueError(f"{source}:{index}: score must be binary, found {score!r}")
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}:{index}: metadata must be an object")
        metadata_scenario_id = metadata.get("scenario_id")
        if metadata_scenario_id is not None and metadata_scenario_id != task_id:
            raise ValueError(f"{source}:{index}: metadata scenario_id changed")
        run_number = metadata.get("run_number")
        if isinstance(run_number, bool) or not isinstance(run_number, int):
            raise ValueError(f"{source}:{index}: run_number must be an integer")
        if not 1 <= run_number <= num_repeats:
            raise ValueError(f"{source}:{index}: unexpected run_number {run_number}")
        if run_number in logical_runs[task_id]:
            raise ValueError(
                f"{source}:{index}: duplicate rollout ({task_id}, {run_number})"
            )
        counts[task_id] += 1
        logical_runs[task_id].add(run_number)

    expected_runs = set(range(1, num_repeats + 1))
    if counts != Counter({scenario_id: num_repeats for scenario_id in expected_ids}):
        raise ValueError(f"{source}: rollout coverage does not match the pinned tasks")
    if any(run_numbers != expected_runs for run_numbers in logical_runs.values()):
        raise ValueError(f"{source}: logical rollout numbers are incomplete")


def summarize_result(
    experiment: Experiment,
    directory: Path,
    *,
    source_scenario_ids: Sequence[str],
    selected_scenario_ids: Sequence[str],
    require_completion: bool,
    num_repeats: int = COMPARISON_NUM_REPEATS,
) -> dict[str, Any]:
    marker = directory / ".eval_done.json"
    if require_completion and not marker.is_file():
        raise FileNotFoundError(f"Completed GAIA2 marker is missing: {marker}")
    output = directory / "output.jsonl"
    rows = _read_jsonl(output)
    _validate_rollouts(
        rows,
        scenario_ids=source_scenario_ids,
        num_repeats=num_repeats,
        source=output,
    )

    selected_ids = set(selected_scenario_ids)
    selected = [row for row in rows if row["task_id"] in selected_ids]
    _validate_rollouts(
        selected,
        scenario_ids=selected_scenario_ids,
        num_repeats=num_repeats,
        source=output,
    )
    score_sum = sum(float(row["score"]) for row in selected)
    scores_by_task: dict[str, list[float]] = {
        scenario_id: [] for scenario_id in selected_scenario_ids
    }
    for row in selected:
        scores_by_task[row["task_id"]].append(float(row["score"]))
    passed_tasks = len(
        [scores for scores in scores_by_task.values() if any(scores)]
    )
    consistently_passed_tasks = len(
        [scores for scores in scores_by_task.values() if all(scores)]
    )
    expected_selected_rows = len(selected_scenario_ids) * num_repeats
    pass_at_1 = score_sum / expected_selected_rows
    pass_at_n = passed_tasks / len(selected_scenario_ids)
    pass_pow_n = consistently_passed_tasks / len(selected_scenario_ids)
    source_metadata: dict[str, Any] = {
        "output_jsonl": str(output),
        "output_jsonl_sha256": sha256_file(output),
    }
    if marker.is_file():
        source_metadata.update(
            {
                "completion_marker": str(marker),
                "completion_marker_sha256": sha256_file(marker),
            }
        )
    return {
        "experiment": experiment.name,
        "source": source_metadata,
        "metrics": {
            "scenario_count": len(selected_scenario_ids),
            "rollout_rows": expected_selected_rows,
            "passed_rollouts": int(score_sum),
            "rollout_success_rate": pass_at_1,
            "passed_tasks": passed_tasks,
            "task_pass_at_3": pass_at_n,
            "pass_at_1": pass_at_1,
            "pass_at_3": pass_at_n,
            "pass_pow_3": pass_pow_n,
        },
    }


def collect_baseline_summaries() -> list[dict[str, Any]]:
    full_ids = partition_scenario_ids("full")
    test_ids = partition_scenario_ids("test")
    if len(full_ids) != SCENARIO_COUNT or len(test_ids) != TEST_SCENARIO_COUNT:
        raise ValueError("Pinned GAIA2 split cardinality changed")
    summaries = []
    for name in COMPARISON_BASELINE_EXPERIMENT_NAMES:
        experiment = get_experiment(name)
        summaries.append(
            summarize_result(
                experiment,
                output_dir(
                    experiment,
                    COMPARISON_NUM_REPEATS,
                    partition="full",
                ),
                source_scenario_ids=full_ids,
                selected_scenario_ids=test_ids,
                require_completion=True,
            )
        )
    return summaries


def build_heldout_comparison(
    candidate: Experiment,
    candidate_directory: Path,
    *,
    baselines: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    test_ids = partition_scenario_ids("test")
    if len(test_ids) != TEST_SCENARIO_COUNT:
        raise ValueError("Pinned GAIA2 test cardinality changed")
    candidate_summary = summarize_result(
        candidate,
        candidate_directory,
        source_scenario_ids=test_ids,
        selected_scenario_ids=test_ids,
        require_completion=False,
    )
    return {
        "schema_version": 1,
        "split_manifest": {
            "name": SPLIT_MANIFEST_NAME,
            "sha256": SPLIT_MANIFEST_SHA256,
        },
        "partition": "test",
        "scenario_count": TEST_SCENARIO_COUNT,
        "num_repeats": COMPARISON_NUM_REPEATS,
        "baselines": list(
            baselines if baselines is not None else collect_baseline_summaries()
        ),
        "candidate": candidate_summary,
    }
