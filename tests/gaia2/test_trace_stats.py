"""Unit coverage for the GAIA2 cost statistics.

The arithmetic here is easy; what is not easy is the bookkeeping the manager's
sequential tool calls force on us, and the three joins that reach across
sidecar, LangGraph log and vLLM log. Those are what these tests pin down.
"""

from __future__ import annotations

import json

from evals.gaia2.trace_stats import (
    Lifecycle,
    SYSTEMS,
    resolve_run,
    join_tool_calls,
    parse_langgraph_log,
    parse_vllm_throughput,
    structural_parallelism,
    temporal_parallelism,
)


def _ai(name: str) -> dict:
    return {"type": "ai", "data": {"tool_calls": [{"name": name}], "content": ""}}


def _tool(payload) -> dict:
    return {"type": "tool", "data": {"content": json.dumps(payload)}}


def _spawn(run_id: str) -> list[dict]:
    return [_ai("run"), _tool({"subagent_run_id": run_id})]


def _wait(*run_ids: str) -> list[dict]:
    return [
        _ai("wait"),
        _tool([{"subagent_run_id": r, "status": "responded"} for r in run_ids]),
    ]


def test_consecutive_spawns_are_the_only_source_of_parallelism():
    messages = [*_spawn("a"), *_spawn("b"), *_spawn("c"), *_wait("a", "b", "c")]

    stats = structural_parallelism(messages)

    assert stats["max_outstanding"] == 3
    assert stats["spawn_bursts"] == [3]
    assert stats["max_reports_per_wait"] == 3
    assert stats["unclosed_at_end"] == 0


def test_new_and_fork_start_no_runs_and_legacy_spawns_still_count():
    messages = [
        _ai("new"), _tool({"subagent_id": "s1"}),
        _ai("fork"), _tool({"subagent_id": "s2"}),
        *_spawn("a"),
        _ai("spawn_subagent"), _tool({"subagent_run_id": "b"}),
        *_wait("a", "b"),
    ]

    stats = structural_parallelism(messages)

    assert stats["spawn_bursts"] == [2]
    assert stats["max_outstanding"] == 2
    assert stats["spawn_errors"] == 0


def test_alternating_spawn_and_wait_never_batches():
    messages = [*_spawn("a"), *_wait("a"), *_spawn("b"), *_wait("b")]

    stats = structural_parallelism(messages)

    assert stats["max_outstanding"] == 1
    assert stats["spawn_bursts"] == [1, 1]
    assert stats["mean_reports_per_wait"] == 1.0


def test_a_wait_that_collects_only_some_reports_leaves_the_rest_outstanding():
    messages = [*_spawn("a"), *_spawn("b"), *_wait("a"), *_spawn("c")]

    stats = structural_parallelism(messages)

    # `b` was never reported, so `c` starts with two already in flight.
    assert stats["max_outstanding"] == 2
    assert stats["unclosed_at_end"] == 2


def test_wait_with_nothing_running_is_counted_not_crashed():
    messages = [_ai("wait"), {"type": "tool", "data": {"content": "No active runs remain."}}]

    stats = structural_parallelism(messages)

    assert stats["empty_waits"] == 1
    assert stats["waits"] == 0


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


def test_identical_tool_calls_are_disambiguated_by_the_langgraph_interval():
    call = {"id": "1", "name": "Files__open", "args": {"path": "/a"}}
    subagents = [
        {"subagent_run_id": "early", "tool_calls": [dict(call)]},
        {"subagent_run_id": "late", "tool_calls": [dict(call)]},
    ]
    env_calls = [
        {"tool": "Files__open", "arguments": {"path": "/a"}, "started_at": 5.0, "result": "first"},
        {"tool": "Files__open", "arguments": {"path": "/a"}, "started_at": 50.0, "result": "second"},
    ]
    lifecycles = {
        "early": Lifecycle("early", "success", 0.0, 0.0, 10.0, 10.0, 0.0),
        "late": Lifecycle("late", "success", 40.0, 40.0, 60.0, 20.0, 0.0),
    }

    results, quality = join_tool_calls(subagents, env_calls, lifecycles)

    assert results["early"] == ["first"]
    assert results["late"] == ["second"]
    assert quality == {"unique": 2, "ambiguous": 0, "unmatched": 0}


def test_a_tool_call_with_no_env_record_is_reported_not_silently_dropped():
    subagents = [{"subagent_run_id": "x", "tool_calls": [{"name": "Gone", "args": {}}]}]

    results, quality = join_tool_calls(subagents, [], {})

    assert results["x"] == [None]
    assert quality["unmatched"] == 1


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


def test_throughput_integral_charges_each_rate_over_the_preceding_gap(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text(
        "INFO 09-06 04:17:05 [loggers.py:273] Engine 000: Avg prompt throughput: "
        "100.0 tokens/s, Avg generation throughput: 10.0 tokens/s, "
        "Prefix cache hit rate: 50.0\n"
        "INFO 09-06 04:17:15 [loggers.py:273] Engine 000: Avg prompt throughput: "
        "200.0 tokens/s, Avg generation throughput: 20.0 tokens/s, "
        "Prefix cache hit rate: 80.0\n"
    )

    total = parse_vllm_throughput(log, year=2026)

    # Only the second sample has a preceding gap: 200 * 10s and 20 * 10s.
    assert total.prompt_tokens == 2000.0
    assert total.generation_tokens == 200.0
    assert total.prefix_hit_rate == 0.8
    # 80% of the served prefix came from cache, so the logical context was 5x.
    assert round(total.logical_prompt_tokens) == 10000


def test_a_restart_gap_does_not_invent_tokens(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text(
        "INFO 09-06 04:00:00 [loggers.py:273] Engine 000: Avg prompt throughput: "
        "100.0 tokens/s, Avg generation throughput: 10.0 tokens/s\n"
        "INFO 09-06 06:00:00 [loggers.py:273] Engine 000: Avg prompt throughput: "
        "100.0 tokens/s, Avg generation throughput: 10.0 tokens/s\n"
    )

    total = parse_vllm_throughput(log, year=2026)

    assert total.prompt_tokens == 0.0
    assert total.seconds == 0.0


def test_an_almost_total_cache_hit_rate_is_refused_not_amplified(tmp_path):
    # At 99.8% the amplification is 500x, so the last reported digit decides the
    # answer. Reporting it would have put 41.9M tokens per rollout in a table.
    log = tmp_path / "vllm.log"
    log.write_text(
        "INFO 09-06 04:00:00 [loggers.py:273] Engine 000: Avg prompt throughput: "
        "100.0 tokens/s, Avg generation throughput: 10.0 tokens/s, "
        "Prefix cache hit rate: 99.8\n"
        "INFO 09-06 04:00:10 [loggers.py:273] Engine 000: Avg prompt throughput: "
        "100.0 tokens/s, Avg generation throughput: 10.0 tokens/s, "
        "Prefix cache hit rate: 99.8\n"
    )

    total = parse_vllm_throughput(log, year=2026)

    assert total.prefix_hit_rate == 0.998
    assert total.prompt_tokens == 1000.0
    assert total.logical_prompt_tokens is None


def _usage(**details) -> dict:
    return {"type": "ai", "data": {"tool_calls": [], "content": "", "usage_metadata": details}}


def test_reasoning_and_cached_tokens_are_kept_when_the_provider_reports_them():
    # A remote reasoning API fills in detail a local non-thinking vLLM omits:
    # on DeepSeek roughly half the output is reasoning and most of the input is
    # served from the provider's cache, which is the whole cost story for it.
    from evals.gaia2.trace_stats import structural_parallelism

    message = _usage(
        input_tokens=2629,
        output_tokens=792,
        input_token_details={"cache_read": 2304},
        output_token_details={"reasoning": 507},
    )

    detail = message["data"]["usage_metadata"]
    assert detail["output_token_details"]["reasoning"] < detail["output_tokens"]
    assert detail["input_token_details"]["cache_read"] < detail["input_tokens"]
    # An AI message carrying no tool call must not disturb the spawn/wait replay.
    assert structural_parallelism([message])["max_outstanding"] == 0


def test_deepseek_runs_resolve_by_their_own_name(tmp_path):
    domain = tmp_path / "search"
    domain.mkdir()
    (domain / "deepseek-v4-flash-0731-reasoning-max-teacher-gemma4-e4b-"
     "non-thinking-n3-prompt-teacher").mkdir()
    (domain / "deepseek-v4-flash-0731-reasoning-max-teacher-gemma4-e4b-"
     "non-thinking-n3-prompt-teacher.INVALID-superseded").mkdir()

    found = resolve_run("DeepSeek teacher", "e4b", "search", tmp_path)

    assert found is not None and "INVALID" not in found.name


def test_every_system_the_report_names_has_a_resolver():
    assert "DeepSeek teacher" in SYSTEMS
