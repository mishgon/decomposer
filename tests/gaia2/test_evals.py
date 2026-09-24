from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.common import METRICS_FILENAME
from evals.gaia2 import comparison
from evals.gaia2 import metrics as metrics_module
from evals.gaia2 import run as eval_run
from evals.gaia2 import submit
from gyms.gaia2 import run_eval as launcher
from gyms.gaia2.experiments import ALL_EXPERIMENTS
from gyms.gaia2.partition import partition_scenario_ids

HELDOUT = {"purpose": "evaluation", "domain": "execution", "partition": "test", "limit": None, "num_repeats": 3}


def _write_run(directory: Path, scores: dict[str, list[float]], **marker) -> Path:
    directory.mkdir(parents=True)
    rows = [
        {"task_id": task, "score": score, "metadata": {"run_number": run}}
        for task, task_scores in scores.items()
        for run, score in enumerate(task_scores, 1)
    ]
    (directory / "output.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / ".eval_done.json").write_text(json.dumps({"state": "complete", **marker}))
    return directory


def test_comparison_applies_only_to_full_heldout_execution_runs():
    assert comparison.applies(HELDOUT)
    assert not comparison.applies({**HELDOUT, "partition": "full"})
    assert not comparison.applies({**HELDOUT, "limit": 5})
    assert not comparison.applies({**HELDOUT, "domain": "search"})
    assert not comparison.applies({**HELDOUT, "purpose": "trace-generation"})


def test_metrics_cover_the_pinned_partition(tmp_path):
    first, second = partition_scenario_ids("test")[:2]
    run_dir = _write_run(
        tmp_path / "run",
        {first: [1.0, 0.0, 1.0], second: [0.0, 0.0, 0.0]},
        experiment="candidate",
        **{**HELDOUT, "limit": 2},
    )

    result = metrics_module.compute(run_dir)

    assert result["metrics"]["pass_at_1"] == 2 / 6
    assert result["metrics"]["pass_at_k"] == 0.5
    assert result["metrics"]["pass_pow_k"] == 0.0
    assert result["gym_metrics"]["fixed_denominator_score"] == 2 / 6


def test_metrics_only_writes_eval_metrics(tmp_path):
    first = partition_scenario_ids("test")[0]
    run_dir = _write_run(tmp_path / "run", {first: [1.0, 1.0, 0.0]}, experiment="candidate", **{**HELDOUT, "limit": 1})

    assert eval_run.main(["--metrics-only", str(run_dir)]) == 0

    assert json.loads((run_dir / METRICS_FILENAME).read_text())["metrics"]["passed_rollouts"] == 2
    assert not (run_dir / eval_run.COMPARISON_FILENAME).exists()


def test_trace_generation_is_not_an_evaluation():
    with pytest.raises(SystemExit, match="trace generation"):
        eval_run.main(["--experiment", ALL_EXPERIMENTS[0].name, "--purpose", "trace-generation", "--dry"])


def test_submitted_evaluations_run_the_evals_entrypoint():
    experiment = ALL_EXPERIMENTS[0]
    staged = Path("/staged")

    default = launcher.build_job_script(staged, experiment, 3, None, force=False)
    evaluated = launcher.build_job_script(staged, experiment, 3, None, force=False, entrypoint=submit.ENTRYPOINT)

    assert "/staged/gyms/gaia2/run.py" in default
    assert default.replace("gyms/gaia2/run.py", "evals/gaia2/run.py") == evaluated
