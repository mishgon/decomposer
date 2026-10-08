from __future__ import annotations

import json
from pathlib import Path

import pytest

from decomposer.prompt_profiles import resolve_decomposer_system_prompt
from decomposer.prompts import DECOMPOSER_SYSTEM_PROMPT
from sft.chat_tools import build_decomposer_chat_tools
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


def _model_call(role: str, started_at: float, task: str, **overrides: object) -> dict:
    return {
        "role": role,
        "started_at": started_at,
        "generation": {"model_name": "Qwen/Qwen3.8-Flash-Next-NVFP4"},
        "tools": TOOLS if role == "decomposer" else [],
        "messages": [{"type": "human", "content": task}],
        "system_message": {
            "type": "system",
            "content": resolve_decomposer_system_prompt("teacher"),
        },
        **overrides,
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
    _write_json(
        execution / "trace.json",
        {
            "messages": [
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
            ],
            "decomposer_agent_runs": [{"prompt": task}],
            "agents": {"a": {"agent_id": "a", "agent_type_id": RESEARCHER}},
            "agent_runs": {},
        },
    )
    later = {"tools": [], "system_message": {"content": "changed later"}}
    for name, call in (
        ("0-researcher", _model_call("researcher", 5.0, task)),
        ("1-later", _model_call("decomposer", 9.0, task, **later)),
        (
            "2-first",
            _model_call(
                "decomposer",
                1.0,
                task,
                **(
                    {"system_message": {"content": system_prompt}}
                    if system_prompt is not None
                    else {}
                ),
            ),
        ),
    ):
        _write_json(execution / "model_calls" / f"{name}.json", call)
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
    return SourceSpec(
        id="wideseek-collection",
        adapter="wideseek",
        path=path,
        benchmark="wideseek",
        environment="wideseek",
        partition="train",
        teacher="qwen38-flash-non-thinking",
        trace_format="wideseek_langgraph_v1",
    )


def _read(path: Path):
    return read_wideseek_source(
        _source(path), SELECTION, system_prompt=DECOMPOSER_SYSTEM_PROMPT
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
    assert first.messages[0]["content"] == DECOMPOSER_SYSTEM_PROMPT
    assert first.attributes["decomposer_model"] == "Qwen/Qwen3.8-Flash-Next-NVFP4"
    # The quoted agent type the core rejected stays in the parallel batch.
    batch = first.messages[2]["tool_calls"]
    assert [call["function"]["arguments"]["agent_type_id"] for call in batch] == [
        RESEARCHER,
        f'"{RESEARCHER}"',
    ]


def test_wideseek_requires_the_teacher_prompt(tmp_path: Path) -> None:
    _attempt(tmp_path, "width-0001", system_prompt="An older prompt.")
    with pytest.raises(ValueError, match="not the teacher prompt"):
        _read(tmp_path)


def test_wideseek_snapshot_takes_each_result_trace_and_first_manager_call(
    tmp_path: Path,
) -> None:
    source = tmp_path / "collection"
    _collection(source)

    directory, manifest = create_snapshot(
        "wideseek", source, tmp_path / "snapshots", snapshot_files(source)
    )

    assert len(manifest["files"]) == 18
    assert all("stale" not in path for path in manifest["files"])
    assert sorted(
        path.rsplit("/", 1)[1] for path in manifest["files"] if "model_calls" in path
    ) == ["2-first.json"] * 6
    assert [record.id for record in _read(directory).records] == [
        record.id for record in _read(source).records
    ]
