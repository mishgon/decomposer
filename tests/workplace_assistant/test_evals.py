from __future__ import annotations

import json
from pathlib import Path

from evals.common import METRICS_FILENAME
from evals.workplace_assistant import comparison
from evals.workplace_assistant import metrics as metrics_module
from evals.workplace_assistant import run as eval_run
from evals.workplace_assistant import submit
from gyms.workplace_assistant import experiments
from gyms.workplace_assistant import run as runner
from gyms.workplace_assistant import run_eval as launcher

CANDIDATE_MARKER = {
    "experiment": comparison.CANDIDATE_EXPERIMENT,
    "purpose": "trace-generation",
    "split": "validation",
    "num_repeats": 3,
    "limit": None,
}


def _write_run(directory: Path, rewards: list[float], marker: dict | None) -> Path:
    directory.mkdir(parents=True)
    rows = [
        {"_ng_task_index": task, "_ng_rollout_index": rollout, "reward": rewards[task * 3 + rollout]}
        for task in range(2)
        for rollout in range(3)
    ]
    (directory / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "rollouts_aggregate_metrics.json").write_text(json.dumps({"mean/reward": 0.5}))
    if marker is not None:
        (directory / ".eval_done.json").write_text(json.dumps({"state": "complete", **marker}))
    return directory


def _prepared(name: str) -> None:
    manifest = experiments.preparation_manifest("validation", name)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"datasets": {"decomposer": {"sha256": "same-dataset"}}}))


def test_comparison_applies_only_to_the_full_candidate_teacher_run():
    assert comparison.applies(CANDIDATE_MARKER)
    assert not comparison.applies({**CANDIDATE_MARKER, "purpose": "evaluation"})
    assert not comparison.applies({**CANDIDATE_MARKER, "limit": 5})
    assert not comparison.applies({**CANDIDATE_MARKER, "experiment": comparison.BASELINE_EXPERIMENT})


def test_metrics_cover_the_validated_grid(tmp_path, monkeypatch):
    monkeypatch.setitem(experiments.SPLIT_ROWS, "validation", 2)
    run_dir = _write_run(tmp_path / "run", [1, 0, 0, 1, 1, 1], {**CANDIDATE_MARKER, "purpose": "evaluation"})

    result = metrics_module.compute(run_dir)

    assert result["metrics"]["pass_at_1"] == 4 / 6
    assert result["metrics"]["pass_at_k"] == 1.0
    assert result["metrics"]["pass_pow_k"] == 0.5
    assert result["gym_aggregate_metrics"] == {"mean/reward": 0.5}
    assert result["integrity"]["rollout_rows"] == 6


def test_metrics_only_writes_the_teacher_comparison_for_the_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(experiments, "DATA_DIR", tmp_path / "data")
    monkeypatch.setitem(experiments.SPLIT_ROWS, "validation", 2)
    baseline_dir = experiments.output_dir(
        experiments.get_experiment(comparison.BASELINE_EXPERIMENT), "validation", 3, purpose="trace-generation"
    )
    _write_run(baseline_dir, [1, 0, 0, 0, 0, 0], {"experiment": comparison.BASELINE_EXPERIMENT})
    candidate_dir = _write_run(tmp_path / "candidate", [1, 1, 0, 0, 0, 1], CANDIDATE_MARKER)
    _prepared(comparison.BASELINE_EXPERIMENT)
    _prepared(comparison.CANDIDATE_EXPERIMENT)

    assert eval_run.main(["--metrics-only", str(candidate_dir)]) == 0

    compared = json.loads((candidate_dir / eval_run.COMPARISON_FILENAME).read_text())
    assert compared["paired_rollout_outcomes"] == {"both_correct": 1, "both_incorrect": 3, "qwen_only": 2}
    assert compared["candidate"]["source"]["completion_marker"].startswith(str(candidate_dir))
    assert json.loads((candidate_dir / METRICS_FILENAME).read_text())["metrics"]["passed_rollouts"] == 3


def test_evaluation_is_the_default_purpose(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "main", lambda argv: seen.append(list(argv)) or 0)

    assert eval_run.main(["--experiment", "any-experiment", "--dry"]) == 0
    assert seen == [["--purpose", "evaluation", "--experiment", "any-experiment", "--dry"]]

    seen.clear()
    assert eval_run.main(["--purpose", "trace-generation", "--experiment", "any-experiment", "--dry"]) == 0
    assert seen[0][:2] == ["--purpose", "trace-generation"]


def test_submitted_evaluations_run_the_evals_entrypoint():
    experiment = experiments.get_experiment(comparison.CANDIDATE_EXPERIMENT)
    staged = Path("/staged")

    default = launcher.build_job_script(staged, experiment, "validation", 3, None, purpose="evaluation", force=False)
    evaluated = launcher.build_job_script(
        staged, experiment, "validation", 3, None, purpose="evaluation", force=False, entrypoint=submit.ENTRYPOINT
    )

    assert "/staged/gyms/workplace_assistant/run.py" in default
    assert "/staged/evals/workplace_assistant/run.py" in evaluated
    assert default.replace("gyms/workplace_assistant", "evals/workplace_assistant") == evaluated
