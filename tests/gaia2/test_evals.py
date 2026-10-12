from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.common import METRICS_FILENAME
from evals.gaia2 import metrics as metrics_module
from evals.gaia2 import run as eval_run

EVALUATION = {
    "purpose": "evaluation",
    "domain": "execution",
    "limit": None,
    "num_repeats": 3,
}


def _write_run(directory: Path, scores: dict[str, list[float]], **marker) -> Path:
    directory.mkdir(parents=True)
    rows = [
        {"task_id": task, "score": score, "metadata": {"run_number": run}}
        for task, task_scores in scores.items()
        for run, score in enumerate(task_scores, 1)
    ]
    (directory / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (directory / ".eval_done.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "scenario_ids": list(scores),
                "harness": "decomposer",
                "model_id": "model",
                **marker,
            }
        )
    )
    return directory


def test_metrics_cover_selected_tasks(tmp_path):
    first, second = ("first", "second")
    run_dir = _write_run(
        tmp_path / "run",
        {first: [1.0, 0.0, 1.0], second: [0.0, 0.0, 0.0]},
        experiment="candidate",
        **{**EVALUATION, "limit": 2},
    )

    result = metrics_module.compute(run_dir)

    assert result["metrics"]["pass_at_1"] == 2 / 6
    assert result["metrics"]["pass_at_k"] == 0.5
    assert result["metrics"]["pass_pow_k"] == 0.0
    assert result["gym_metrics"]["fixed_denominator_score"] == 2 / 6


def test_metrics_only_writes_eval_metrics(tmp_path):
    first = "first"
    run_dir = _write_run(
        tmp_path / "run",
        {first: [1.0, 1.0, 0.0]},
        experiment="candidate",
        **{**EVALUATION, "limit": 1},
    )

    assert eval_run.main(["--metrics-only", str(run_dir)]) == 0

    assert (
        json.loads((run_dir / METRICS_FILENAME).read_text())["metrics"][
            "passed_rollouts"
        ]
        == 2
    )


def test_metrics_use_explicit_task_selection(tmp_path):
    last = "last"
    directory = _write_run(
        tmp_path / "run",
        {last: [1.0, 0.0, 0.0]},
        scenario_ids=[last],
        harness="decomposer",
        model_id="model",
        **EVALUATION,
    )
    result = metrics_module.compute(directory)
    assert result["metrics"]["pass_at_1"] == 1 / 3
    assert result["gym_metrics"]["task_rows"] == 1


@pytest.mark.parametrize("agent", ["decomposer", "qwen_3_5_4b_thinking"])
def test_evaluation_runs_both_domains_and_combines_metrics(
    tmp_path, monkeypatch, agent
):
    calls = []
    monkeypatch.setattr(
        eval_run.runner, "selected_scenario_ids", lambda args: ("same-id",)
    )
    monkeypatch.setattr(eval_run.runner, "validate_preparation", lambda **kwargs: {})

    def execute(args):
        calls.append((args.agent, args.domain, args.concurrency))
        return _write_run(
            args.output_dir,
            {"same-id": [1.0, 0.0] if args.domain == "execution" else [0.0, 0.0]},
            agent=args.agent,
            domain=args.domain,
            num_repeats=args.num_repeats,
        )

    monkeypatch.setattr(eval_run.runner, "execute", execute)
    directory = tmp_path / "eval"
    assert (
        eval_run.main(
            [
                "--agent",
                agent,
                "--num-repeats",
                "2",
                "--concurrency",
                "4",
                "--output-dir",
                str(directory),
            ]
        )
        == 0
    )
    assert calls == [(agent, "execution", 4), (agent, "search", 4)]
    result = json.loads((directory / METRICS_FILENAME).read_text())
    assert result["metrics"]["tasks"] == 2
    assert result["metrics"]["pass_at_1"] == 0.25
    assert result["metrics"]["pass_at_k"] == 0.5
    assert result["errors"] == 0
    assert eval_run.main(["--metrics-only", str(directory)]) == 0


def test_incomplete_suite_cannot_report_success(tmp_path):
    (tmp_path / "evaluation.json").write_text(
        json.dumps({"agent": "decomposer", "domains": ["execution", "search"]})
    )
    _write_run(
        tmp_path / "execution", {"task": [1.0]}, domain="execution", num_repeats=1
    )
    with pytest.raises(FileNotFoundError):
        metrics_module.compute_suite(tmp_path)


def test_eval_dry_run_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_run.runner, "selected_scenario_ids", lambda args: ("task",)
    )
    calls = []
    monkeypatch.setattr(
        eval_run.runner, "execute", lambda args: calls.append(args.domain)
    )
    directory = tmp_path / "dry"
    eval_run.main(["--dry", "--output-dir", str(directory)])
    assert calls == ["execution", "search"]
    assert not directory.exists()
