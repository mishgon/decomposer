from __future__ import annotations

import json
from pathlib import Path

import pytest

from decomposer.prompts import DECOMPOSER_SYSTEM_PROMPT
from sft.chat_tools import build_decomposer_chat_tools, decomposer_system_prompt
from sft.schema import SelectionSpec, SourceSpec
from sft.snapshots import create_snapshot
from sft.wideseek.adapter import read_wideseek_source, snapshot_files

RESEARCHER = "lmrouter/qwen_3_5_4b_unlooped_thinking"
TOOLS = build_decomposer_chat_tools(
    [
        {
            "agent_type_id": RESEARCHER,
            "description": "Qwen3.5-4B unlooped thinking researcher.",
            "assistant_id": RESEARCHER,
        }
    ]
)
SELECTION = SelectionSpec(policy="collector_qualifies", success_threshold=0.9)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _ai(content: str, *calls: dict) -> dict:
    return {"type": "ai", "content": content, "tool_calls": list(calls)}


def _call(name: str, call_id: str, **arguments: str) -> dict:
    return {"name": name, "args": arguments, "id": call_id, "type": "tool_call"}


def _tool(name: str, call_id: str, content: str) -> dict:
    return {"type": "tool", "name": name, "tool_call_id": call_id, "content": content}


def _dump(message: dict) -> dict:
    """A LangChain ``message_to_dict`` dump, the form the gym writes messages in."""
    return {"type": message["type"], "data": message}


def _first_call(role: str, call_id: str, task: str, system_prompt: str) -> dict:
    """The logged start of a thread's first model call, the only one with tools."""
    return {
        "call_id": call_id,
        "role": role,
        "thread_id": f"thread-{role}",
        "started_at": "2026-10-11T12:00:00+00:00",
        "max_tokens": 4096,
        "generation": {"model_name": "Qwen/Qwen3.8-Flash-Next-NVFP4"},
        "request_message_count": 1,
        "request_delta": [_dump({"type": "human", "content": task})],
        "tools": TOOLS if role == "decomposer" else [],
        "system_message": _dump({"type": "system", "content": system_prompt}),
        "status": "started",
    }


def _attempt(
    root: Path,
    task_id: str,
    *,
    status: str = "finished",
    score: float = 1.0,
    evaluation_status: str = "scored",
    cleanup_errors: list[str] | None = None,
    system_prompt: str | None = None,
) -> Path:
    """Write one attempt in the layout of a WideSeek collection."""
    task = f"Build the table for {task_id}."
    attempt_dir = root / "decomposer" / task_id / "attempt-001"
    execution = attempt_dir / f"execution-{task_id[-4:]}abc"
    # A restarted execution that no result names.
    _write_json(attempt_dir / "execution-stale" / "trace.json", {"messages": []})
    messages = [
        {"type": "human", "content": task},
        _ai(
            "Two parts.",
            _call("new", "new-a", agent_type_id=RESEARCHER),
            _call("new", "new-b", agent_type_id=f'"{RESEARCHER}"'),
        ),
        _tool("new", "new-a", '{"agent_id": "a"}'),
        _tool("new", "new-b", f'Unknown agent type ID `"{RESEARCHER}"`.'),
        _ai("", _call("run", "run-a", agent_id="a", prompt="Find rows.")),
        _tool("run", "run-a", '{"agent_run_id": "run-a"}'),
        _ai("", _call("wait", "wait")),
        _tool("wait", "wait", '[{"agent_id": "a", "status": "responded"}]'),
        _ai("| a | b |"),
    ]
    _write_json(
        execution / "trace.json",
        {
            "messages": [_dump(message) for message in messages],
            "decomposer_agent_runs": [{"prompt": task}],
            "agents": {"a": {"agent_id": "a", "agent_type_id": RESEARCHER}},
            "agent_runs": {},
        },
    )
    calls = [
        # A researcher's first call carries tools and a system message too.
        _first_call("researcher", "r-1", "Find rows.", "You are a researcher."),
        _first_call(
            "decomposer", "d-1", task, system_prompt or decomposer_system_prompt()
        ),
        {"call_id": "d-1", "role": "decomposer", "status": "success", "response": []},
        # Later calls log only the new messages.
        {
            "call_id": "d-2",
            "role": "decomposer",
            "status": "started",
            "request_delta": [_dump(messages[1]), _dump(messages[2])],
        },
        {"call_id": "r-1", "role": "researcher", "status": "success", "response": []},
    ]
    (execution / "model_calls.jsonl").write_text(
        "".join(json.dumps(call) + "\n" for call in calls), encoding="utf-8"
    )
    _write_json(
        attempt_dir / "result.json",
        {
            "task_id": task_id,
            "mode": "decomposer",
            "attempt": 1,
            "execution_directory": execution.name,
            "status": status,
            "revision": "d8d33ca",
            "evaluation": {
                "status": evaluation_status,
                "metric": "item_f1",
                "score": score,
            },
            **({"cleanup_errors": cleanup_errors} if cleanup_errors else {}),
        },
    )
    return execution


def _source(path: Path) -> SourceSpec:
    results = len(list(path.glob("decomposer/*/attempt-*/result.json")))
    return SourceSpec(
        id="wideseek-collection",
        adapter="wideseek",
        # Only the builder resolves snapshots; these tests read the fixture.
        snapshot="sha256:" + "0" * 64,
        expected_native_rollouts=results,
        expected_candidates=results,
        benchmark="wideseek",
        environment="wideseek",
        partition="train",
        teacher="qwen38-flash-non-thinking",
        trace_format="wideseek_langgraph_v2",
    )


def _read(path: Path):
    return read_wideseek_source(
        _source(path),
        SELECTION,
        source_dir=path,
        system_prompt=decomposer_system_prompt(),
    )


def _collection(root: Path) -> None:
    _attempt(root, "width-0001")
    _attempt(root, "width-0002", score=0.95)
    _attempt(root, "width-0003", score=0.9)
    _attempt(root, "width-0004", status="timeout")
    _attempt(root, "width-0005", cleanup_errors=["sandbox"])
    _attempt(root, "width-0006", evaluation_status="judge_error")


def test_wideseek_keeps_what_the_collector_counts_as_success(tmp_path: Path) -> None:
    _collection(tmp_path)

    result = _read(tmp_path)

    assert result.counts["rollouts"] == 6
    assert result.counts["excluded_reward"] == 4
    assert [record.source.task_id for record in result.records] == [
        "width-0001",
        "width-0002",
    ]
    first, second = result.records
    assert first.group_id == "wideseek:width-0001"
    assert first.source.rollout_id == "attempt-001/execution-0001abc"
    assert (first.outcome.reward, second.outcome.reward) == (1.0, 0.95)
    # The tools and prompt come from the manager's first call, not a later one.
    assert first.tools == TOOLS
    assert first.messages[0]["content"] == decomposer_system_prompt()
    assert first.attributes["decomposer_model"] == "Qwen/Qwen3.8-Flash-Next-NVFP4"
    # The quoted agent type the core rejected stays in the parallel batch.
    batch = first.messages[2]["tool_calls"]
    assert [call["function"]["arguments"]["agent_type_id"] for call in batch] == [
        RESEARCHER,
        f'"{RESEARCHER}"',
    ]


@pytest.mark.parametrize("logged", ["An older prompt.", DECOMPOSER_SYSTEM_PROMPT])
def test_wideseek_requires_the_teacher_prompt(tmp_path: Path, logged: str) -> None:
    _attempt(tmp_path, "width-0001", system_prompt=logged)
    with pytest.raises(ValueError, match="not the teacher prompt"):
        _read(tmp_path)


def test_wideseek_snapshot_takes_each_result_trace_and_call_log(
    tmp_path: Path,
) -> None:
    source = tmp_path / "collection"
    _collection(source)

    directory, manifest = create_snapshot(
        "wideseek", source, tmp_path / "snapshots", snapshot_files(source)
    )

    assert len(manifest["files"]) == 18
    assert all("stale" not in path for path in manifest["files"])
    assert sum(path.endswith("/model_calls.jsonl") for path in manifest["files"]) == 6
    assert [record.id for record in _read(directory).records] == [
        record.id for record in _read(source).records
    ]
