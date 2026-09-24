from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.common import METRICS_FILENAME, binary_pass_metrics
from evals.tau2_gym import metrics as metrics_module
from evals.tau2_gym import run as eval_run
from gyms.tau2_gym import run as runner
from gyms.tau2_gym.experiments import EXPERIMENTS_BY_NAME


def _row(task: int, repeat: int, reward: float, domain: str) -> dict:
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "reward": reward,
        "domain": domain,
        "task_id": f"{domain}-{task}",
        "response": {
            "output": [
                {"type": "function_call", "name": "spawn_subagent", "call_id": "s1",
                 "arguments": json.dumps({"subagent_type_id": "worker", "prompt": "do it"})},
                {"type": "function_call_output", "call_id": "s1", "output": json.dumps({"subagent_run_id": "r1"})},
                {"type": "function_call", "name": "wait", "call_id": "w1", "arguments": "{}"},
            ]
        },
    }


def _completed_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_row(0, 0, 1.0, "airline"), _row(0, 1, 0.0, "airline"),
            _row(1, 0, 1.0, "retail"), _row(1, 1, 1.0, "retail")]
    (run_dir / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    marker = {
        "run_name": "exp-pool-n2",
        "experiment": {"name": "exp"},
        "pool": "pool",
        "num_repeats": 2,
        "dataset_rows": 2,
        "limit": None,
    }
    marker["result"] = runner.validate_result(run_dir / "rollouts.jsonl", expected_tasks=2, num_repeats=2)
    (run_dir / ".eval_done.json").write_text(json.dumps(marker))
    return run_dir


def test_binary_pass_metrics_counts_rollouts_and_tasks():
    result = binary_pass_metrics({0: [1.0, 0.0], 1: [1.0, 1.0]})

    assert result["tasks"] == 2
    assert result["repeats"] == 2
    assert result["pass_at_1"] == 0.75
    assert result["pass_at_k"] == 1.0
    assert result["pass_pow_k"] == 0.5


def test_metrics_agree_with_the_gym_run_summary(tmp_path):
    run_dir = _completed_run(tmp_path)

    result = metrics_module.compute(run_dir)

    marker = json.loads((run_dir / ".eval_done.json").read_text())
    assert result["metrics"]["pass_at_1"] == marker["result"]["pass_rate"]
    assert result["by_domain"]["airline"]["pass_at_1"] == 0.5
    assert result["by_domain"]["retail"]["pass_pow_k"] == 1.0
    assert result["decomposition"]["spawns_per_rollout"]["mean"] == 1.0


def test_metrics_require_a_complete_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="not complete"):
        metrics_module.compute(tmp_path)


def test_metrics_only_writes_eval_metrics_unless_told_not_to(tmp_path):
    run_dir = _completed_run(tmp_path)

    assert eval_run.main(["--metrics-only", str(run_dir), "--no-write"]) == 0
    assert not (run_dir / METRICS_FILENAME).exists()

    assert eval_run.main(["--metrics-only", str(run_dir)]) == 0
    written = json.loads((run_dir / METRICS_FILENAME).read_text())
    assert written["metrics"]["pass_at_1"] == 0.75


def test_complete_runs_are_not_run_again(tmp_path, monkeypatch):
    run_dir = _completed_run(tmp_path)
    monkeypatch.setattr(runner, "resolve_output_dir", lambda args: ("exp-pool-n2", run_dir))

    def fail(args):
        raise AssertionError("a complete run must not be executed again")

    monkeypatch.setattr(runner, "execute", fail)

    assert eval_run.main(["--experiment", next(iter(EXPERIMENTS_BY_NAME))]) == 0
    assert (run_dir / METRICS_FILENAME).is_file()


def test_resolve_output_dir_matches_the_run_name(tmp_path):
    name = next(iter(EXPERIMENTS_BY_NAME))
    args = runner.build_parser().parse_args(["--experiment", name, "--num-repeats", "3"])

    run_name, output_dir = runner.resolve_output_dir(args)

    assert run_name.endswith("-n3")
    assert output_dir == runner.RESULTS_ROOT / run_name
    args = runner.build_parser().parse_args(["--experiment", name, "--output-dir", str(tmp_path)])
    assert runner.resolve_output_dir(args)[1] == tmp_path
