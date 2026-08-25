from __future__ import annotations

import hashlib
import io
import json
import tarfile
from copy import deepcopy
from pathlib import Path

import pytest

from data.sft.adapters.toolathlon_gym import read_toolathlon_gym_source
from data.sft.builder import LoadedBuildSpec, prepare_dataset
from data.sft.import_toolathlon import import_archive, sha256_file
from data.sft.schema import (
    BuildSpec,
    DatasetIdentity,
    PolicySpec,
    SelectionSpec,
    SourceSpec,
    SplitSpec,
)
from decomposer.prompts import DECOMPOSER_SYSTEM_PROMPT

RUN_ID = "20260824T101524Z-203eac76"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "spawn_subagent",
            "description": "Spawn one Toolathlon subagent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subagent_type_id": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["subagent_type_id", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for Toolathlon subagent reports.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _human(content: str) -> dict:
    return {"type": "human", "data": {"type": "human", "content": content}}


def _ai(
    content: str,
    *,
    calls: list[dict] | None = None,
    reasoning: str | None = None,
    invalid_calls: list[dict] | None = None,
) -> dict:
    return {
        "type": "ai",
        "data": {
            "type": "ai",
            "content": content,
            "tool_calls": calls or [],
            "invalid_tool_calls": invalid_calls or [],
            "additional_kwargs": {"reasoning_content": reasoning or ""},
        },
    }


def _tool(name: str, call_id: str, content: str) -> dict:
    return {
        "type": "tool",
        "data": {
            "type": "tool",
            "content": content,
            "name": name,
            "tool_call_id": call_id,
        },
    }


def _trace(
    task: str, *, passed: bool = True, tools: object = TOOLS
) -> tuple[dict, dict, dict]:
    episode_id = f"{RUN_ID}-{hashlib.sha256(task.encode()).hexdigest()[:8]}-r001-a001"
    prompt = f"Complete Toolathlon task {task}."
    spawn_a = {
        "name": "spawn_subagent",
        "args": {"subagent_type_id": "worker", "prompt": "Do part A."},
        "id": f"{episode_id}-spawn-a",
        "type": "tool_call",
    }
    spawn_b = {
        "name": "spawn_subagent",
        "args": {"subagent_type_id": "worker", "prompt": "Do part B."},
        "id": f"{episode_id}-spawn-b",
        "type": "tool_call",
    }
    wait = {
        "name": "wait",
        "args": {},
        "id": f"{episode_id}-wait",
        "type": "tool_call",
    }
    trace = {
        "schema_version": 2,
        "episode_id": episode_id,
        "run_id": RUN_ID,
        "task": task,
        "repetition": 1,
        "attempt": 1,
        "purpose": "trace-generation",
        "decomposer_model": "deepseek/deepseek-v4-flash-0731",
        "subagent_model": "google/gemma-4-26B-A4B-it",
        "tools": deepcopy(tools),
        "messages": [
            _human(prompt),
            _ai(
                "Delegate both parts.",
                calls=[spawn_a, spawn_b],
                reasoning="The parts are independent.",
            ),
            _tool("spawn_subagent", spawn_b["id"], '{"subagent_run_id":"b"}'),
            _tool("spawn_subagent", spawn_a["id"], '{"subagent_run_id":"a"}'),
            _ai("Wait for both.", calls=[wait], reasoning="Collect the reports."),
            _tool(
                "wait",
                wait["id"],
                '[{"subagent_run_id":"a","status":"success","content":"A"},'
                '{"subagent_run_id":"b","status":"error","content":"retryable"}]',
            ),
            _ai("The requested task is complete.", reasoning="Report the result."),
        ],
        "subagent_runs": {
            "a": {"subagent_run_id": "a", "status": "success", "report": {}},
            "b": {"subagent_run_id": "b", "status": "error", "report": {}},
        },
    }
    runtime = {
        "task_config": {
            "id": task,
            "task_str": prompt,
            "needed_mcp_servers": ["canvas", "excel"],
        }
    }
    result = {
        "episode_id": episode_id,
        "task": task,
        "pass": passed,
        "returncode": 0 if passed else 1,
    }
    return trace, runtime, result


def _source(root: Path, episodes: list[tuple[dict, dict, dict | None]]) -> Path:
    run_manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "running",
        "episodes": [
            {
                "task": trace["task"],
                "repetition": trace["repetition"],
                "status": "completed",
                "attempts": [
                    (
                        {"attempt": 1, "status": "failed"}
                        if index == 0
                        else {"attempt": 1, "status": "completed"}
                    )
                ],
            }
            for index, (trace, _runtime, _result) in enumerate(episodes)
        ],
    }
    _write_json(root / "runs" / RUN_ID / "manifest.json", run_manifest)
    for trace, runtime, result in episodes:
        task = trace["task"]
        episode_id = trace["episode_id"]
        trace_dir = root / "traces" / task / episode_id
        _write_json(trace_dir / "trace.json", trace)
        _write_json(trace_dir / "runtime.json", runtime)
        if result is not None:
            _write_json(root / "evals" / task / episode_id / "result.json", result)

    files = {}
    for path in sorted(root.rglob("*.json")):
        if path.name == "import_manifest.json":
            continue
        content = path.read_bytes()
        files[path.relative_to(root).as_posix()] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    _write_json(
        root / "import_manifest.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "archive": {"sha256": "a" * 64},
            "files": files,
        },
    )
    return root


def _source_spec(source: Path) -> SourceSpec:
    return SourceSpec(
        id="toolathlon-source",
        adapter="toolathlon_gym",
        path=source,
        benchmark="toolathlon_gym",
        environment="toolathlon_gym",
        partition="train",
        teacher="deepseek-v4-flash-0731",
    )


def _add_tar_json(archive: tarfile.TarFile, name: str, value: object) -> None:
    content = json.dumps(value).encode()
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def test_import_archive_selects_one_run_and_is_idempotent(tmp_path: Path) -> None:
    trace, runtime, result = _trace("alpha")
    archive_path = tmp_path / "toolathlon.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        _add_tar_json(
            archive,
            f"toolathlon_gym/runs/{RUN_ID}/manifest.json",
            {"schema_version": 1, "run_id": RUN_ID, "episodes": []},
        )
        episode_id = trace["episode_id"]
        base = f"toolathlon_gym/traces/alpha/{episode_id}"
        _add_tar_json(archive, f"{base}/trace.json", trace)
        _add_tar_json(archive, f"{base}/runtime.json", runtime)
        _add_tar_json(
            archive,
            f"toolathlon_gym/evals/alpha/{episode_id}/result.json",
            result,
        )
        _add_tar_json(archive, f"{base}/workspace/ignored.json", {"large": True})
        _add_tar_json(
            archive,
            "toolathlon_gym/traces/other/other-run-episode/trace.json",
            {"run_id": "other-run"},
        )

    expected = sha256_file(archive_path)
    destination = import_archive(
        archive_path, RUN_ID, tmp_path / "imports", expected_sha256=expected
    )
    repeated = import_archive(
        archive_path, RUN_ID, tmp_path / "imports", expected_sha256=expected
    )

    assert repeated == destination
    manifest = json.loads((destination / "import_manifest.json").read_text())
    assert manifest["counts"] == {"traces": 1, "runtimes": 1, "evaluations": 1}
    assert not list(destination.rglob("workspace"))
    assert not list(destination.rglob("other-run-episode"))


@pytest.mark.parametrize("unsafe_kind", ["traversal", "symlink"])
def test_import_archive_rejects_unsafe_members(
    tmp_path: Path, unsafe_kind: str
) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        if unsafe_kind == "traversal":
            _add_tar_json(archive, "toolathlon_gym/../escape.json", {})
        else:
            info = tarfile.TarInfo("toolathlon_gym/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp/target"
            archive.addfile(info)

    with pytest.raises(ValueError, match="Unsafe|Unsupported"):
        import_archive(archive_path, RUN_ID, tmp_path / "imports")


def test_toolathlon_adapter_filters_and_normalizes_schema_v2_traces(
    tmp_path: Path,
) -> None:
    success = _trace("success")
    failed_reward = _trace("failed-reward", passed=False)
    missing_tools = list(_trace("missing-tools"))
    missing_tools[0] = {**missing_tools[0], "tools": None}
    missing_result = _trace("missing-result")
    source = _source(
        tmp_path / "source",
        [success, failed_reward, tuple(missing_tools), (*missing_result[:2], None)],
    )

    result = read_toolathlon_gym_source(
        _source_spec(source),
        SelectionSpec(invalid_policy="exclude"),
        system_prompt=DECOMPOSER_SYSTEM_PROMPT,
    )

    assert len(result.records) == 1
    assert result.counts["rollouts"] == 4
    assert result.counts["eligible"] == 1
    assert result.counts["excluded_reward"] == 1
    assert result.counts["excluded_invalid_tool_schema"] == 1
    assert result.counts["excluded_missing_evaluation"] == 1
    record = result.records[0]
    assert record.messages[0] == {
        "role": "system",
        "content": DECOMPOSER_SYSTEM_PROMPT,
    }
    assert record.tools == TOOLS
    assert record.attributes["subagent_statuses"] == {"error": 1, "success": 1}
    assert record.attributes["parallel_spawn_normalization"] == {
        "messages": 1,
        "tool_calls": 2,
    }
    spawn_messages = [
        message
        for message in record.messages
        if message.get("role") == "assistant"
        and message.get("tool_calls")
        and message["tool_calls"][0]["function"]["name"] == "spawn_subagent"
    ]
    assert len(spawn_messages) == 2
    assert spawn_messages[0]["teacher_reasoning"] == "The parts are independent."
    assert "teacher_reasoning" not in spawn_messages[1]
    assert result.source_manifest["paired_records"] == 3
    assert result.source_manifest["unpaired_trace_records"] == 1
    assert result.source_manifest["sidecar_failure_records"] == 1
    assert result.source_manifest["tool_schema_origin"] == "trace.tools"


def test_toolathlon_adapter_errors_on_missing_tools_in_strict_mode(
    tmp_path: Path,
) -> None:
    trace, runtime, result = _trace("missing-tools", tools=None)
    source = _source(tmp_path / "source", [(trace, runtime, result)])

    with pytest.raises(ValueError, match="trace.tools must be a list"):
        read_toolathlon_gym_source(
            _source_spec(source),
            SelectionSpec(invalid_policy="error"),
            system_prompt=DECOMPOSER_SYSTEM_PROMPT,
        )


def test_toolathlon_adapter_rejects_untracked_import_files(tmp_path: Path) -> None:
    source = _source(tmp_path / "source", [_trace("alpha")])
    _write_json(source / "traces" / "unexpected" / "trace.json", {})

    with pytest.raises(ValueError, match="file set changed"):
        read_toolathlon_gym_source(
            _source_spec(source),
            SelectionSpec(invalid_policy="exclude"),
            system_prompt=DECOMPOSER_SYSTEM_PROMPT,
        )


def test_canonical_builder_accepts_toolathlon_source(tmp_path: Path) -> None:
    source = _source(
        tmp_path / "source", [_trace(f"task-{index}") for index in range(10)]
    )
    spec = BuildSpec(
        spec_version=1,
        dataset=DatasetIdentity(id="toolathlon-test", version="v1"),
        policy=PolicySpec(id="decomposer-default"),
        sources=(_source_spec(source),),
        selection=SelectionSpec(),
        split=SplitSpec(strategy="prompt_fixed", validation_fraction=0.1, seed=42),
    )
    prepared = prepare_dataset(
        LoadedBuildSpec(path=tmp_path / "spec.yaml", sha256="1" * 64, spec=spec),
        tmp_path / "datasets",
        git_revision="test-revision",
        require_clean_git=False,
    )
    train = [json.loads(line) for line in prepared.train_path.read_text().splitlines()]
    validation = [
        json.loads(line) for line in prepared.validation_path.read_text().splitlines()
    ]

    assert len(train) == 9
    assert len(validation) == 1
    assert {row["group_id"] for row in train}.isdisjoint(
        row["group_id"] for row in validation
    )
    assert all(
        row["messages"][0]["content"] == DECOMPOSER_SYSTEM_PROMPT for row in train
    )
    assert prepared.manifest["preparation"]["adapter_versions"] == {"toolathlon_gym": 1}
    assert prepared.manifest["normalization"] == {
        "strategy": "parallel_spawn_calls_to_single_call_turns",
        "traces": 10,
        "messages": 10,
        "tool_calls": 20,
    }


def test_canonical_builder_requires_one_tool_schema(tmp_path: Path) -> None:
    changed_tools = deepcopy(TOOLS)
    changed_tools[0]["function"]["description"] = "A different teacher schema."
    source = _source(
        tmp_path / "source",
        [_trace("alpha"), _trace("beta", tools=changed_tools)],
    )
    spec = BuildSpec(
        spec_version=1,
        dataset=DatasetIdentity(id="toolathlon-test", version="v1"),
        policy=PolicySpec(id="decomposer-default"),
        sources=(_source_spec(source),),
        selection=SelectionSpec(),
        split=SplitSpec(strategy="prompt_fixed", validation_fraction=0.5, seed=42),
    )

    with pytest.raises(ValueError, match="Expected one consistent tool schema"):
        prepare_dataset(
            LoadedBuildSpec(path=tmp_path / "spec.yaml", sha256="1" * 64, spec=spec),
            tmp_path / "datasets",
            git_revision="test-revision",
            require_clean_git=False,
        )
