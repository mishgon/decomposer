import json
from math import isclose

import pytest

from evals.toolathlon_bench import run as evaluation, summary, trace_stats
from gyms.toolathlon_bench import run as bench


def write_run(root, outcomes, repetitions=3):
    """outcomes maps task -> list of (returncode, timed_out, pass or None for no result)."""
    episodes = []
    for task, attempts in outcomes.items():
        for repetition, (returncode, timed_out, passed) in enumerate(attempts, start=1):
            episode_id = f"run-{task}-r{repetition:03d}"
            episodes.append({"task": task, "repetition": repetition, "episode_id": episode_id,
                             "returncode": returncode, "timed_out": timed_out})
            if passed is not None:
                result = root / "evals" / task / episode_id / "result.json"
                result.parent.mkdir(parents=True)
                result.write_text(json.dumps({"pass": passed}))
    (root / "manifest.json").write_text(json.dumps(
        {"tasks": list(outcomes), "repetitions": repetitions, "episodes": episodes}
    ))


def test_summary_counts_only_clean_native_passes(tmp_path):
    write_run(tmp_path, {
        "solved": [(0, False, True), (0, False, True), (0, False, True)],
        # One pass, one native fail, one pass from an agent that crashed.
        "flaky": [(0, False, True), (0, False, False), (1, False, True)],
        # A timeout, an unscored attempt, and an attempt missing from the manifest.
        "broken": [(-1, True, True), (0, False, None)],
    })
    result = summary.summarize(tmp_path)
    assert result["successes_by_task"] == {"solved": 3, "flaky": 1, "broken": 0}
    assert result["expected_attempts"] == 9 and result["scored_attempts"] == 7
    assert result["successful_attempts"] == 4
    metrics = result["metrics"]
    assert isclose(metrics["pass@1"], (1 + 1 / 3 + 0) / 3)
    assert isclose(metrics["pass@3"], (1 + 1 + 0) / 3)
    assert isclose(metrics["pass^3"], (1 + 0 + 0) / 3)
    assert "pass@5" not in metrics


def test_summary_denominator_counts_unselected_tasks_as_failures(tmp_path):
    write_run(tmp_path, {"solved": [(0, False, True)]}, repetitions=1)
    assert summary.summarize(tmp_path, 108)["metrics"] == {"pass@1": 1 / 108}
    with pytest.raises(ValueError):
        summary.summarize(tmp_path, 0)


def test_summary_cli_prints_metrics(tmp_path, monkeypatch, capsys):
    write_run(tmp_path, {"solved": [(0, False, True)]}, repetitions=1)
    monkeypatch.setattr("sys.argv", ["summary", str(tmp_path), "--denominator", "2"])
    summary.main()
    assert json.loads(capsys.readouterr().out)["metrics"] == {"pass@1": 0.5}


def test_evaluation_runs_benchmark_and_saves_summary(tmp_path, monkeypatch):
    seen = {}

    def run(args):
        seen["args"] = args
        root = tmp_path / "run"
        root.mkdir()
        write_run(root, {"find-alita-paper": [(0, False, True)]}, repetitions=1)
        return root

    monkeypatch.setattr(evaluation, "run", run)
    root = evaluation.main(["find-alita-paper", "--denominator", "4"])
    args = seen["args"]
    assert args.purpose == "evaluation"
    assert args.output_dir == bench.REPO_ROOT / "artifacts/evals/toolathlon_bench"
    assert not hasattr(args, "denominator")
    saved = json.loads((root / "summary.json").read_text())
    assert saved["denominator"] == 4 and saved["metrics"] == {"pass@1": 0.25}


def test_trace_stats_totals_traces_and_usage(tmp_path, capsys, monkeypatch):
    first = tmp_path / "traces/a/e1"
    first.mkdir(parents=True)
    (first / "trace.json").write_text(json.dumps({
        "agent_error": None, "agent_runs": {"r1": {}, "r2": {}},
        "decomposer_model": "lmrouter/flash",
        "started_at": "2026-10-04T12:00:00+00:00", "finished_at": "2026-10-04T12:01:00+00:00",
    }))
    (first / "usage.json").write_text(json.dumps({"totals": {"input_tokens": 10, "model_responses": 2}}))
    second = tmp_path / "traces/b/e2"
    second.mkdir(parents=True)
    (second / "trace.json").write_text(json.dumps({
        "agent_error": "RuntimeError()", "agent_runs": {}, "decomposer_model": None,
        "agent_api_model": "Qwen/Qwen3.5-4B-unlooped",
    }))
    broken = tmp_path / "traces/c/e3"
    broken.mkdir(parents=True)
    (broken / "trace.json").write_text("{")

    stats = trace_stats.summarize_traces(tmp_path)
    assert stats["totals"] == {"traces": 2, "agent_errors": 1, "agent_runs": 2,
                               "input_tokens": 10, "output_tokens": 0, "reasoning_tokens": 0,
                               "cache_read_tokens": 0, "model_responses": 2}
    assert stats["models"] == {"lmrouter/flash": 1, "Qwen/Qwen3.5-4B-unlooped": 1}
    assert stats["mean_episode_seconds"] == 60 and stats["timed_episodes"] == 1
    assert [item["path"] for item in stats["unreadable"]] == [str(broken / "trace.json")]

    monkeypatch.setattr("sys.argv", ["trace_stats", str(tmp_path)])
    trace_stats.main()
    assert json.loads(capsys.readouterr().out)["totals"]["traces"] == 2
