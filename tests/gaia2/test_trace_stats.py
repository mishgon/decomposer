from __future__ import annotations

import json

import pytest

from evals.gaia2 import trace_stats
from evals.gaia2.trace_stats import parse_langgraph_log, temporal_parallelism


def test_temporal_parallelism_is_time_weighted_over_the_busy_span():
    # Two subagents overlap for 1s of a 3s busy span: (1*1 + 2*1 + 1*1) / 3.
    stats = temporal_parallelism([(0.0, 2.0), (1.0, 3.0)])

    assert stats["max_concurrent"] == 2
    assert stats["busy_union_seconds"] == 3.0
    assert stats["busy_sum_seconds"] == 4.0
    assert stats["mean_concurrent"] == round(4.0 / 3.0, 3)
    assert stats["overlap_factor"] == round(4.0 / 3.0, 3)


def test_subagent_without_an_end_timestamp_is_excluded_not_extrapolated():
    stats = temporal_parallelism([(0.0, 2.0), (1.0, None)])

    assert stats["measured_subagents"] == 1
    assert stats["max_concurrent"] == 1


def test_langgraph_lifecycle_survives_the_ansi_colour_codes(tmp_path):
    log = tmp_path / "langgraph.log"
    log.write_text(
        "\x1b[2m2026-09-06T01:17:04.308968Z\x1b[0m [\x1b[32minfo\x1b[0m] "
        "\x1b[1mBackground run succeeded\x1b[0m run_created_at=2026-09-06T01:16:58.889228+00:00 "
        "run_ended_at=2026-09-06T01:17:04.308767+00:00 run_exec_ms=4455 "
        "run_id=01a0744a-4272-73e3-96a5-f51317cf7d1e "
        "run_started_at=2026-09-06T01:16:59.853191+00:00 run_wait_time_ms=963\n"
    )

    parsed = parse_langgraph_log(log)

    entry = parsed["01a0744a-4272-73e3-96a5-f51317cf7d1e"]
    assert entry.status == "succeeded"
    assert entry.wait_seconds == 0.963
    assert round(entry.ended - entry.started, 3) == 4.456


def test_single_repeat_scores_join_without_run_numbers(tmp_path):
    rows = [
        {"task_id": "a", "score": 1.0, "metadata": {"scenario_id": "a"}},
        {
            "task_id": "b",
            "score": 0.0,
            "metadata": {"scenario_id": "b", "run_number": 2},
        },
    ]
    (tmp_path / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assert trace_stats._outcomes(tmp_path) == {("a", None): 1.0, ("b", 2): 0.0}


def test_cumulative_usage_and_runs_are_counted_once(tmp_path):
    sidecars = tmp_path / "decomposer_sidecars"
    sidecars.mkdir()
    (sidecars / "task.json").write_text(
        json.dumps(
            {
                "scenario_id": "task",
                "run_number": 2,
                "turns": [
                    {
                        "manager": {
                            "usage": {"output_tokens": tokens},
                            "subagent_states": [
                                {"agent_run_id": "worker", "status": "responded"}
                            ],
                            "timing": {"elapsed_seconds": 3},
                        }
                    }
                    for tokens in (10, 25)
                ],
            }
        )
    )
    result = trace_stats.analyse(tmp_path)
    assert result["summary"]["manager_observed_usage"] == {"output_tokens": 25}
    assert result["summary"]["worker_usage"] is None
    record = result["rollouts"][0]
    assert record["subagent_statuses"] == {"responded": 1}
    assert record["manager_turn_seconds"] == 6
    assert record["score"] is None


def test_missing_usage_is_not_zero_and_failure_is_preserved(tmp_path):
    sidecars = tmp_path / "decomposer_sidecars"
    sidecars.mkdir()
    (sidecars / "task.json").write_text(
        json.dumps(
            {
                "scenario_id": "task",
                "turns": [
                    {"manager": {"failure": {"error": "failed", "traceback": "trace"}}}
                ],
            }
        )
    )
    result = trace_stats.analyse(tmp_path)
    assert result["summary"]["manager_observed_usage"] is None
    assert result["summary"]["manager_usage_complete_rollouts"] == 0
    assert result["rollouts"][0]["manager_failures"] == [
        {"error": "failed", "traceback": "trace"}
    ]


@pytest.mark.parametrize("run_number", [None, 2])
def test_lite_duration_and_usage_join_by_repeat(tmp_path, run_number):
    (tmp_path / "output.jsonl").write_text(
        json.dumps(
            {"task_id": "task", "score": 1, "metadata": {"run_number": run_number}}
        )
    )
    lite = tmp_path / "lite"
    lite.mkdir()
    suffix = f"_run_{run_number}" if run_number is not None else ""
    usage = {"agent_0": {"prompt_tokens": [5], "completion_tokens": [2]}}
    (lite / f"task{suffix}_hash.json").write_text(
        json.dumps(
            {
                "scenario_id": "task",
                "run_duration": 12,
                "per_agent_llm_usage_stats": usage,
            }
        )
    )
    record = trace_stats.analyse(tmp_path)["rollouts"][0]
    assert record["run_duration_seconds"] == 12
    assert record["are_llm_usage"] == usage
