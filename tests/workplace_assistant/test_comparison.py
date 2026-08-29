from __future__ import annotations

import json
from pathlib import Path

import pytest

from gyms.workplace_assistant import comparison
from gyms.workplace_assistant import experiments


def _write_source(root: Path, name: str, rewards: list[float], *, complete: bool) -> Path:
    experiment = experiments.get_experiment(name)
    directory = experiments.output_dir(
        experiment,
        "validation",
        3,
        purpose="trace-generation",
    )
    directory.mkdir(parents=True)
    rows = []
    for task in range(2):
        for rollout in range(3):
            rows.append(
                {
                    "_ng_task_index": task,
                    "_ng_rollout_index": rollout,
                    "reward": rewards[task * 3 + rollout],
                }
            )
    (directory / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    if complete:
        (directory / ".eval_done.json").write_text("{}\n", encoding="utf-8")
    manifest = experiments.preparation_manifest("validation", name)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"datasets": {"decomposer": {"sha256": "same-dataset"}}}),
        encoding="utf-8",
    )
    return directory


def test_workplace_teacher_comparison_is_coverage_checked_and_pinned(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(experiments, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(experiments, "DATA_DIR", tmp_path / "data")
    monkeypatch.setitem(experiments.SPLIT_ROWS, "validation", 2)
    baseline = _write_source(
        tmp_path,
        comparison.BASELINE_EXPERIMENT,
        [1, 0, 0, 0, 0, 0],
        complete=True,
    )
    candidate = _write_source(
        tmp_path,
        comparison.CANDIDATE_EXPERIMENT,
        [1, 1, 0, 0, 0, 1],
        complete=False,
    )

    result = comparison.build_teacher_comparison(candidate)

    assert result["scenario_count"] == 2
    assert result["baseline"]["metrics"]["passed_rollouts"] == 1
    assert result["candidate"]["metrics"]["passed_rollouts"] == 3
    assert result["delta_candidate_minus_baseline"][
        "rollout_success_rate"
    ] == pytest.approx(2 / 6)
    assert result["paired_rollout_outcomes"] == {
        "both_correct": 1,
        "both_incorrect": 3,
        "qwen_only": 2,
    }
    assert result["baseline"]["source"]["completion_marker"].startswith(
        str(baseline)
    )


def test_workplace_teacher_comparison_requires_complete_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(experiments, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(experiments, "DATA_DIR", tmp_path / "data")
    monkeypatch.setitem(experiments.SPLIT_ROWS, "validation", 2)
    _write_source(
        tmp_path,
        comparison.BASELINE_EXPERIMENT,
        [0, 0, 0, 0, 0, 0],
        complete=False,
    )
    candidate = _write_source(
        tmp_path,
        comparison.CANDIDATE_EXPERIMENT,
        [0, 0, 0, 0, 0, 0],
        complete=False,
    )

    with pytest.raises(FileNotFoundError, match="Completion marker"):
        comparison.build_teacher_comparison(candidate)
