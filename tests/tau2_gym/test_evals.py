from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.common import METRICS_FILENAME, binary_pass_metrics
from evals.tau2_gym import analyze_traces
from evals.tau2_gym import metrics as metrics_module
from evals.tau2_gym import run as eval_run
from gyms.tau2_gym import run as runner
from gyms.tau2_gym.experiments import EXPERIMENTS_BY_NAME


def _call(call_id: str, name: str, arguments: dict, result) -> list[dict]:
    return [
        {"type": "function_call", "name": name, "call_id": call_id, "arguments": json.dumps(arguments)},
        {"type": "function_call_output", "call_id": call_id, "output": json.dumps(result)},
    ]


def _row(task: int, repeat: int, reward: float, domain: str) -> dict:
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "reward": reward,
        "domain": domain,
        "task_id": f"{domain}-{task}",
        "response": {
            "output": [
                *_call("n1", "new", {"subagent_type_id": "worker"}, {"subagent_id": "a"}),
                *_call("r1", "run", {"subagent_id": "a", "prompt": "do it"}, {"subagent_run_id": "ra"}),
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
    assert result["decomposition"]["runs_per_rollout"]["mean"] == 1.0
    assert result["decomposition"]["subagents_per_rollout"]["mean"] == 1.0


def test_trace_analysis_counts_parallel_runs_forks_and_reused_subagents():
    wait = lambda call_id, run_ids: _call(  # noqa: E731
        call_id, "wait", {}, [{"subagent_run_id": run_id, "status": "responded"} for run_id in run_ids]
    )
    row = {
        "reward": 1.0,
        "response": {"output": [
            *_call("n1", "new", {"subagent_type_id": "worker"}, {"subagent_id": "a"}),
            *_call("n2", "new", {"subagent_type_id": "worker"}, {"subagent_id": "b"}),
            *_call("r1", "run", {"subagent_id": "a", "prompt": "x"}, {"subagent_run_id": "ra1"}),
            *_call("r2", "run", {"subagent_id": "b", "prompt": "y"}, {"subagent_run_id": "rb1"}),
            *wait("w1", ["ra1", "rb1"]),
            *_call("f1", "fork", {"subagent_id": "a"}, {"subagent_id": "c"}),
            *_call("r3", "run", {"subagent_id": "a", "prompt": "more"}, {"subagent_run_id": "ra2"}),
            *wait("w2", ["ra2"]),
        ]},
    }

    record = analyze_traces.analyse_rollout(row)

    assert (record["n_new"], record["n_fork"], record["n_run"], record["n_wait"]) == (2, 1, 3, 2)
    assert record["run_batches"] == [2, 1]
    assert record["max_fanout"] == 2
    assert record["reused_runs"] == 1
    assert record["subagent_types"] == {"worker": 3}
    assert record["responses_in_run_order"] is True


def test_trace_analysis_still_reads_spawn_subagent_rollouts():
    row = {"reward": 0.0, "response": {"output": [
        *_call("s1", "spawn_subagent", {"subagent_type_id": "worker", "prompt": "x"}, {"subagent_run_id": "r1"}),
        *_call("s2", "spawn_subagent", {"subagent_type_id": "worker", "prompt": "y"}, {"subagent_run_id": "r2"}),
        *_call("w1", "wait", {}, [{"subagent_run_id": "r1"}, {"subagent_run_id": "r2"}]),
    ]}}

    record = analyze_traces.analyse_rollout(row)

    assert (record["n_new"], record["n_run"], record["max_fanout"]) == (2, 2, 2)
    assert record["subagent_types"] == {"worker": 2}


def test_scoring_summary_separates_paths_and_counts_unreported_calls(tmp_path):
    rows = [
        {"reward": 1.0, "breakdown": {"scoring": "server_log_v1", "unreported_calls": 3}},
        {"reward": 0.0, "breakdown": {"scoring": "server_log_v1", "unreported_calls": 0}},
        {"reward": 1.0, "breakdown": {"terms": {}}},
        {"reward": 0.0},
    ]

    summary = metrics_module.scoring_summary(rows)

    assert summary["paths"] == {"not_scored": 1, "reported_replay_v0": 1, "server_log_v1": 2}
    assert summary["rollouts_with_unreported_calls"] == 1
    assert summary["unreported_calls"] == 3
    assert summary["pass_at_1_of_rollouts_with_unreported_calls"] == 1.0
    run_dir = _completed_run(tmp_path)
    assert metrics_module.compute(run_dir)["scoring"]["paths"] == {"not_scored": 4}


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
