from __future__ import annotations

import json
from pathlib import Path

import pytest

from gyms.gaia2.comparison import summarize_result
from gyms.gaia2.experiments import SIMPLE_EXPERIMENT


def _write_result(
    directory: Path,
    *,
    score_override: float | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario_id in ("scenario_a", "scenario_b"):
        for run_number in (1, 2, 3):
            score = float(scenario_id == "scenario_b" and run_number < 3)
            if score_override is not None and scenario_id == "scenario_b":
                score = score_override
            rows.append(
                {
                    "task_id": scenario_id,
                    "score": score,
                    "metadata": {
                        "scenario_id": scenario_id,
                        "run_number": run_number,
                    },
                }
            )
    (directory / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )
    (directory / ".eval_done.json").write_text("{}\n", encoding="utf-8")


def test_completed_full_result_is_reused_for_heldout_metrics(tmp_path) -> None:
    _write_result(tmp_path)

    summary = summarize_result(
        SIMPLE_EXPERIMENT,
        tmp_path,
        source_scenario_ids=("scenario_a", "scenario_b"),
        selected_scenario_ids=("scenario_b",),
        require_completion=True,
    )

    assert summary["metrics"] == {
        "scenario_count": 1,
        "rollout_rows": 3,
        "passed_rollouts": 2,
        "rollout_success_rate": pytest.approx(2 / 3),
        "passed_tasks": 1,
        "task_pass_at_3": 1.0,
        "pass_at_1": pytest.approx(2 / 3),
        "pass_at_3": 1.0,
        "pass_pow_3": 0.0,
    }
    assert summary["source"]["output_jsonl_sha256"]
    assert summary["source"]["completion_marker_sha256"]


def test_baseline_reuse_requires_completion_marker(tmp_path) -> None:
    _write_result(tmp_path)
    (tmp_path / ".eval_done.json").unlink()

    with pytest.raises(FileNotFoundError, match="Completed GAIA2 marker is missing"):
        summarize_result(
            SIMPLE_EXPERIMENT,
            tmp_path,
            source_scenario_ids=("scenario_a", "scenario_b"),
            selected_scenario_ids=("scenario_b",),
            require_completion=True,
        )


def test_baseline_reuse_rejects_nonbinary_rewards(tmp_path) -> None:
    _write_result(tmp_path, score_override=0.5)

    with pytest.raises(ValueError, match="score must be binary"):
        summarize_result(
            SIMPLE_EXPERIMENT,
            tmp_path,
            source_scenario_ids=("scenario_a", "scenario_b"),
            selected_scenario_ids=("scenario_b",),
            require_completion=True,
        )


def test_baseline_reuse_rejects_duplicate_logical_rollouts(tmp_path) -> None:
    _write_result(tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "output.jsonl").read_text().splitlines()
    ]
    rows[0]["metadata"]["run_number"] = rows[1]["metadata"]["run_number"]
    (tmp_path / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate rollout"):
        summarize_result(
            SIMPLE_EXPERIMENT,
            tmp_path,
            source_scenario_ids=("scenario_a", "scenario_b"),
            selected_scenario_ids=("scenario_b",),
            require_completion=True,
        )
