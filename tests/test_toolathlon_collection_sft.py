from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from decomposer.prompts import DECOMPOSER_SYSTEM_PROMPT
from sft.adapters.toolathlon_gym import read_toolathlon_gym_source, snapshot_files
from sft.chat_tools import build_decomposer_chat_tools
from sft.schema import (
    BuildSpec,
    DatasetIdentity,
    PolicySpec,
    SelectionSpec,
    SourceSpec,
    SplitSpec,
)
from sft.snapshots import create_snapshot

RUN_ID = "20261005T152306Z-37ef9df8"
AGENT_TYPE = "qwen_3_5_4b_thinking"
AGENT_DESCRIPTION = "Qwen3.5-4B thinking agent equipped with all the available tools."
SELECTION = SelectionSpec(policy="collector_qualifies", success_threshold=0.9)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _ai(content: str, *calls: dict) -> dict:
    return {
        "type": "ai",
        "content": content,
        "tool_calls": list(calls),
        "invalid_tool_calls": [],
        "additional_kwargs": {"refusal": None},
    }


def _call(name: str, call_id: str, **arguments: str) -> dict:
    return {"name": name, "args": arguments, "id": call_id, "type": "tool_call"}


def _tool(name: str, call_id: str, content: str) -> dict:
    return {"type": "tool", "name": name, "tool_call_id": call_id, "content": content}


def _episode(
    root: Path,
    task: str,
    repetition: int,
    *,
    passed: bool,
    checks: tuple[int, int] | None = None,
    agent_error: str | None = None,
    agent_type: str = AGENT_TYPE,
    evaluated: bool = True,
) -> str:
    """Write one episode in the layout of an sft/toolathlon_gym collection."""
    episode_id = f"{RUN_ID}-{task[:8]}-r{repetition:03d}-a001"
    prompt = f"Complete {task}."
    messages = [
        {"type": "human", "content": prompt},
        # A parallel batch with a call the core answered with an error.
        _ai(
            "Start.",
            _call("new", "new-a", agent_type_id=agent_type),
            _call("claim_done", "claim"),
        ),
        _tool("new", "new-a", '{"agent_id": "a"}'),
        _tool("claim_done", "claim", "Error: claim_done is not a valid tool."),
        _ai("", _call("run", "run-a", agent_id="a", prompt="Do it.")),
        _tool("run", "run-a", '{"agent_run_id": "run-a"}'),
        _ai("", _call("wait", "wait")),
        _tool("wait", "wait", '[{"agent_id": "a", "status": "responded"}]'),
        _ai("Done."),
    ]
    trace_dir = root / "traces" / task / episode_id
    _write_json(
        trace_dir / "trace.json",
        {
            "episode_id": episode_id,
            "run_id": RUN_ID,
            "task": task,
            "repetition": repetition,
            "attempt": 1,
            "purpose": "trace-generation",
            "agent_base_url": "http://agents.internal:8000",
            "decomposer_generation_config": {
                "model_name": "Qwen/Qwen3.8-Flash-Next-FP8",
                "openai_api_base": "http://teacher.internal/v1",
            },
            "agent_api_model": "Qwen/Qwen3.5-4B-unlooped",
            "messages": messages,
            "agents": {
                "a": {"agent_id": "a", "agent_type_id": agent_type, "thread_id": "a"}
            },
            "agent_runs": {"run-a": {"agent_run_id": "run-a", "status": "responded"}},
        },
    )
    _write_json(
        trace_dir / "runtime.json",
        {
            "task_config": {
                "id": task,
                "task_str": prompt,
                "needed_mcp_servers": ["excel"],
            },
            "agent_model": {"base_url": "http://agents.internal:8000"},
        },
    )
    if evaluated:
        _write_json(
            root / "evals" / task / episode_id / "result.json",
            {
                "episode_id": episode_id,
                "task": task,
                "pass": passed,
                "native_pass": passed,
                "agent_error": agent_error,
                "returncode": 0,
                "native_result": (
                    {"total_passed": checks[0], "total_checks": checks[1]}
                    if checks
                    else None
                ),
                "stdout": "",
                "stderr": "",
            },
        )
    return episode_id


def _collection(root: Path) -> dict[str, str]:
    return {
        "strict": _episode(root, "alpha-task", 1, passed=True),
        "partial": _episode(root, "beta-task", 1, passed=False, checks=(19, 20)),
        "at_threshold": _episode(
            root, "gamma-task", 1, passed=False, checks=(18, 20)
        ),
        "agent_error": _episode(
            root, "delta-task", 1, passed=True, agent_error="Episode timed out"
        ),
        "running": _episode(root, "epsilon-task", 1, passed=True, evaluated=False),
    }


def _source(path: Path, **overrides: object) -> SourceSpec:
    return SourceSpec(
        **{
            "id": "toolathlon-collection",
            "adapter": "toolathlon_gym",
            "path": path,
            "benchmark": "toolathlon_gym",
            "environment": "toolathlon_gym",
            "partition": "train",
            "teacher": "qwen38-flash-non-thinking",
            "trace_format": "toolathlon_langgraph_v1",
            "native_subagent_types": [
                {"id": AGENT_TYPE, "description": AGENT_DESCRIPTION}
            ],
            **overrides,
        }
    )


def _read(source: SourceSpec):
    return read_toolathlon_gym_source(
        source, SELECTION, system_prompt=DECOMPOSER_SYSTEM_PROMPT
    )


def test_collection_keeps_what_the_collector_counts_as_success(tmp_path: Path) -> None:
    episodes = _collection(tmp_path)

    result = _read(_source(tmp_path))

    # The running episode has no evaluation yet; 18/20 is not above 0.9, and an
    # agent error never qualifies, even with a pass.
    assert result.counts["rollouts"] == 4
    assert result.counts["excluded_reward"] == 2
    assert result.counts["eligible"] == 2
    strict, partial = result.records
    assert strict.id.endswith(episodes["strict"])
    assert partial.id.endswith(episodes["partial"])
    assert strict.group_id == "toolathlon_gym:alpha-task"
    assert (strict.outcome.reward, strict.outcome.metrics["binary_pass"]) == (1.0, 1.0)
    assert partial.outcome.reward == pytest.approx(0.95)
    assert partial.outcome.metrics["binary_pass"] == 0.0
    assert partial.attributes["partial_score_source"] == "native_total"
    assert strict.attributes["decomposer_model"] == "Qwen/Qwen3.8-Flash-Next-FP8"
    assert result.source_manifest["qualifying"] == {
        "strict_pass": 1,
        "score_above_threshold": 1,
    }

    assert strict.tools == build_decomposer_chat_tools(
        [
            {
                "agent_type_id": AGENT_TYPE,
                "description": AGENT_DESCRIPTION,
                "assistant_id": AGENT_TYPE,
            }
        ]
    )
    assert strict.messages[0]["content"] == DECOMPOSER_SYSTEM_PROMPT
    assert strict.messages[1]["content"] == "Complete alpha-task."
    # The batch with the invented call stays as emitted, with the core's error.
    batch = strict.messages[2]["tool_calls"]
    assert [call["function"]["name"] for call in batch] == ["new", "claim_done"]
    assert strict.messages[4]["content"].startswith("Error:")


def test_collection_checks_prompts_and_declared_agent_types(tmp_path: Path) -> None:
    _episode(tmp_path, "alpha-task", 1, passed=True)
    mismatch = _episode(tmp_path, "beta-task", 1, passed=True)
    runtime_path = tmp_path / "traces" / "beta-task" / mismatch / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["task_config"]["task_str"] = "Another task."
    runtime_path.write_text(json.dumps(runtime))

    result = _read(_source(tmp_path))
    assert result.counts["excluded_prompt_mismatch"] == 1
    assert result.counts["eligible"] == 1

    _episode(tmp_path, "gamma-task", 1, passed=True, agent_type="other_type")
    with pytest.raises(ValueError, match="undeclared subagent types: other_type"):
        _read(_source(tmp_path))


def test_collection_snapshot_takes_finished_episodes_and_redacts_endpoints(
    tmp_path: Path,
) -> None:
    source = tmp_path / "collection"
    episodes = _collection(source)

    files = snapshot_files(source)
    directory, manifest = create_snapshot(
        "toolathlon_gym", source, tmp_path / "snapshots", files
    )

    assert len(manifest["files"]) == 12
    assert not any(episodes["running"] in path for path in manifest["files"])
    trace_path = directory / "traces" / "alpha-task" / episodes["strict"]
    trace = json.loads((trace_path / "trace.json").read_text())
    assert "agent_base_url" not in trace
    assert "openai_api_base" not in trace["decomposer_generation_config"]
    assert [record.id for record in _read(_source(directory)).records] == [
        record.id for record in _read(_source(source)).records
    ]


def test_collection_format_requires_its_agent_types_and_selection(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="native_subagent_types"):
        _source(tmp_path, native_subagent_types=[])
    with pytest.raises(ValidationError, match="success_threshold"):
        SelectionSpec(policy="collector_qualifies")

    def spec(source: SourceSpec) -> BuildSpec:
        return BuildSpec(
            spec_version=3,
            dataset=DatasetIdentity(id="collection", version="v1"),
            policy=PolicySpec(id="decomposer-default"),
            sources=(source,),
            selection=SelectionSpec(),
            split=SplitSpec(
                strategy="prompt_fixed", validation_fraction=0.5, seed=42
            ),
        )

    counts = {"expected_native_rollouts": 1, "expected_candidates": 1}
    with pytest.raises(ValidationError, match="select with collector_qualifies"):
        spec(_source(tmp_path, **counts))
    selection = {"policy": "collector_qualifies", "success_threshold": 0.9}
    assert spec(_source(tmp_path, selection=selection, **counts))
