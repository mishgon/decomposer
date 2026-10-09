from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from datasets import Dataset
from pydantic import ValidationError

from sft.adapters.registry import SNAPSHOT_FILES
from sft.builder import LoadedBuildSpec, load_build_spec, prepare_dataset
from sft.snapshots import create_snapshot
from sft.schema import (
    EXCLUSION_REASONS,
    BuildSpec,
    DatasetIdentity,
    PolicySpec,
    SelectionSpec,
    SourceSpec,
    SplitSpec,
    TraceValidationError,
    canonical_json,
    normalize_response_tools,
    sha256_text,
    validate_chat_tools,
    validate_decomposer_messages,
)
from sft.chat_tools import build_decomposer_chat_tools
from decomposer.prompt_profiles import (
    DECOMPOSER_STUDENT_SYSTEM_PROMPT,
    resolve_decomposer_system_prompt,
)
from decomposer.prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    EARLY_RESPONSE_ERROR,
    PARALLEL_WAIT_CALL_ERROR,
)
from sft.train import (
    _validate_dataset_system_prompt_profile,
    _validate_manifest,
)


def _response_tool(name: str, *parameters: str) -> dict:
    return {
        "name": name,
        "description": f"The {name} tool.",
        "parameters": {
            "type": "object",
            "properties": {parameter: {"type": "string"} for parameter in parameters},
            **({"required": list(parameters)} if parameters else {}),
        },
        "strict": False,
        "type": "function",
    }


TOOLS = [
    _response_tool("new", "agent_type_id"),
    _response_tool("fork", "agent_id"),
    _response_tool("run", "agent_id", "prompt"),
    _response_tool("wait"),
]
LEGACY_TOOLS = [
    _response_tool("spawn_subagent", "subagent_type_id", "prompt"),
    _response_tool("wait"),
]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def _input(task_index: int) -> list[dict]:
    return [
        {"role": "system", "type": "message", "content": "Benchmark system."},
        {"role": "user", "type": "message", "content": f"Task {task_index}."},
    ]


def _ai(
    content: str,
    *,
    tool_calls: list[dict] | None = None,
    reasoning: str | None = None,
) -> dict:
    output = []
    if reasoning is not None:
        output.append(
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": reasoning}],
            }
        )
    return {
        "type": "ai",
        "content": content,
        "tool_calls": tool_calls or [],
        "invalid_tool_calls": [],
        "response_metadata": {"nemo_gym_response": {"output": output}},
    }


def _call(name: str, call_id: str, **arguments: str) -> dict:
    return {"name": name, "args": arguments, "id": call_id}


def _result(name: str, call_id: str, content: object) -> dict:
    return {
        "type": "tool",
        "content": content if isinstance(content, str) else json.dumps(content),
        "tool_call_id": call_id,
        "name": name,
    }


def _rollout(
    task_index: int,
    *,
    rollout_index: int = 0,
    reward: float = 1.0,
    prompt_task_index: int | None = None,
) -> dict:
    """A new -> run -> wait -> final-answer Decomposer trajectory."""
    suffix = f"{task_index}-{rollout_index}"
    new_id = f"new-{suffix}"
    run_id = f"run-{suffix}"
    wait_id = f"wait-{suffix}"
    agent_id = f"subagent-{task_index}"
    agent_run_id = f"subagent-run-{task_index}"
    prompt_task_index = task_index if prompt_task_index is None else prompt_task_index
    return {
        "agent_ref": {"type": "responses_api_agents", "name": "decomposer"},
        "responses_create_params": {"input": _input(prompt_task_index)},
        "response": {"tools": deepcopy(TOOLS)},
        "reward": reward,
        "final_state": {
            "messages": [
                {
                    "type": "human",
                    "content": f"Benchmark system.\n\nTask {task_index}.",
                },
                _ai(
                    "",
                    reasoning=f"Delegate task {task_index}.",
                    tool_calls=[_call("new", new_id, agent_type_id="small")],
                ),
                _result("new", new_id, {"agent_id": agent_id}),
                _ai(
                    "",
                    reasoning="Run it.",
                    tool_calls=[
                        _call(
                            "run",
                            run_id,
                            agent_id=agent_id,
                            prompt=f"Do task {task_index}.",
                        )
                    ],
                ),
                _result("run", run_id, {"agent_run_id": agent_run_id}),
                _ai(
                    "",
                    reasoning="Wait.",
                    tool_calls=[_call("wait", wait_id)],
                ),
                _result(
                    "wait",
                    wait_id,
                    [
                        {
                            "agent_id": agent_id,
                            "agent_run_id": agent_run_id,
                            "status": "responded",
                            "response": "Done.",
                            "error": None,
                        }
                    ],
                ),
                _ai("The task is complete.", reasoning="Report success."),
            ]
        },
        "_ng_task_index": task_index,
        "_ng_rollout_index": rollout_index,
    }


def _legacy_rollout(task_index: int) -> dict:
    """A spawn_subagent/wait trajectory of the retired Decomposer core."""
    rollout = _rollout(task_index)
    spawn_id = f"spawn-{task_index}"
    wait_id = f"wait-{task_index}"
    rollout["response"]["tools"] = deepcopy(LEGACY_TOOLS)
    rollout["final_state"]["messages"] = [
        rollout["final_state"]["messages"][0],
        _ai(
            "",
            tool_calls=[
                _call(
                    "spawn_subagent",
                    spawn_id,
                    subagent_type_id="small",
                    prompt=f"Do task {task_index}.",
                )
            ],
        ),
        _result("spawn_subagent", spawn_id, {"subagent_run_id": "run"}),
        _ai("", tool_calls=[_call("wait", wait_id)]),
        _result(
            "wait",
            wait_id,
            [{"subagent_run_id": "run", "status": "success", "content": "Done."}],
        ),
        _ai("The task is complete."),
    ]
    return rollout


def _materialized(
    task_index: int, *, rollout_index: int = 0, prompt_task_index: int | None = None
) -> dict:
    categories = ["email", "calendar", "crm", "analytics", "project"]
    prompt_task_index = task_index if prompt_task_index is None else prompt_task_index
    return {
        "responses_create_params": {"input": _input(prompt_task_index)},
        "category": categories[task_index % len(categories)],
        "environment_name": "workplace",
        "_ng_task_index": task_index,
        "_ng_rollout_index": rollout_index,
    }


def _source(
    root: Path,
    teacher: str,
    rollouts: list[dict] | None = None,
    materialized: list[dict] | None = None,
) -> Path:
    source = root / teacher
    if rollouts is None:
        rollouts = [_rollout(index) for index in range(10)] + [_rollout(10, reward=0.0)]
    if materialized is None:
        materialized = [_materialized(index) for index in range(11)]
    _write_jsonl(source / "rollouts.jsonl", rollouts)
    _write_jsonl(source / "rollouts_materialized_inputs.jsonl", materialized)
    _write_jsonl(source / "rollouts_failures.jsonl", [{"failure": "synthetic"}])
    return source


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _prepare_fixture_dataset(
    source_dirs: list[Path],
    output_dir: Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    success_reward: float = 1.0,
    invalid_policy: str = "exclude",
    max_traces_per_prompt_per_teacher: int | None = None,
    version: str = "v3",
    system_prompt_profile: str | None = None,
):
    """Test helper that exercises the new canonical builder without Git state."""
    spec = BuildSpec(
        spec_version=1,
        dataset=DatasetIdentity(id=output_dir.name, version=version),
        policy=PolicySpec(
            id="decomposer-default",
            system_prompt_profile=system_prompt_profile,
        ),
        sources=tuple(
            SourceSpec(
                id=source.name,
                adapter="nemo_gym",
                path=source,
                benchmark="workplace_assistant",
                environment="workplace",
                partition="train",
                teacher=source.name,
            )
            for source in source_dirs
        ),
        selection=SelectionSpec(
            success_reward=success_reward,
            invalid_policy=invalid_policy,
            max_traces_per_prompt_per_teacher=max_traces_per_prompt_per_teacher,
        ),
        split=SplitSpec(
            strategy="prompt_fixed",
            validation_fraction=validation_fraction,
            seed=seed,
        ),
    )
    return prepare_dataset(
        LoadedBuildSpec(
            path=output_dir / "spec.yaml",
            sha256="0" * 64,
            spec=spec,
        ),
        output_dir.parent,
        git_revision="test-revision",
        require_clean_git=False,
    )


def test_source_paths_can_be_required_and_overridden_explicitly(
    tmp_path: Path,
) -> None:
    source_dir = _source(tmp_path, "teacher")
    spec = BuildSpec(
        spec_version=1,
        dataset=DatasetIdentity(id="override-fixture", version="v1"),
        policy=PolicySpec(id="decomposer-default"),
        sources=(
            SourceSpec(
                id="workplace",
                adapter="nemo_gym",
                benchmark="workplace_assistant",
                environment="workplace",
                partition="train",
                teacher="teacher",
            ),
        ),
        selection=SelectionSpec(),
        split=SplitSpec(
            strategy="prompt_fixed",
            validation_fraction=0.1,
            seed=42,
        ),
    )
    loaded = LoadedBuildSpec(
        path=tmp_path / "spec.yaml",
        sha256="0" * 64,
        spec=spec,
    )
    with pytest.raises(ValueError, match="require explicit path overrides"):
        prepare_dataset(
            loaded,
            tmp_path / "missing",
            git_revision="test",
            require_clean_git=False,
        )
    with pytest.raises(ValueError, match="Unknown source path override"):
        prepare_dataset(
            loaded,
            tmp_path / "unknown",
            git_revision="test",
            require_clean_git=False,
            source_paths={"unknown": source_dir},
        )

    prepared = prepare_dataset(
        loaded,
        tmp_path / "prepared-overrides",
        git_revision="test",
        require_clean_git=False,
        source_paths={"workplace": source_dir},
    )
    assert prepared.manifest["records"]["total"] == 10
    assert prepared.manifest["records"]["train"] == 9
    assert prepared.manifest["records"]["validation"] == 1
    assert prepared.manifest["sources"][0]["locator"] == str(source_dir.resolve())


def test_prepare_groups_teacher_variants_and_writes_manifest_v3(tmp_path: Path) -> None:
    prepared = _prepare_fixture_dataset(
        [_source(tmp_path, "teacher-a"), _source(tmp_path, "teacher-b")],
        tmp_path / "prepared",
        validation_fraction=0.2,
    )
    train = _read_jsonl(prepared.train_path)
    validation = _read_jsonl(prepared.validation_path)
    assert len(train) == 16
    assert len(validation) == 4
    assert {record["group_id"] for record in train}.isdisjoint(
        record["group_id"] for record in validation
    )
    assert prepared.manifest["format_version"] == 3
    assert prepared.manifest["canonical_schema_version"] == 1
    assert len(prepared.manifest["dataset"]["fingerprint"]) == 64
    assert prepared.manifest["split"]["train_groups"] == 8
    assert prepared.manifest["split"]["validation_groups"] == 2
    filtering = prepared.manifest["filtering"]
    assert filtering["rollouts"] == 22
    assert filtering["eligible_before_cap"] == 20
    assert filtering["included"] == 20
    assert filtering["excluded_reward"] == 2
    assert filtering["sidecar_failure_records"] == 2
    assert all(reason in filtering for reason in EXCLUSION_REASONS)

    example = train[0]
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][0]["content"] == DECOMPOSER_SYSTEM_PROMPT
    assert example["messages"][0]["content"] != DECOMPOSER_STUDENT_SYSTEM_PROMPT
    assert example["messages"][1]["role"] == "user"
    assert example["messages"][-1]["teacher_reasoning"] == "Report success."
    assert [
        message["tool_calls"][0]["function"]["name"]
        for message in example["messages"]
        if message.get("tool_calls")
    ] == ["new", "run", "wait"]
    assert [tool["function"]["name"] for tool in example["tools"]] == [
        "new",
        "fork",
        "run",
        "wait",
    ]
    assert example["source"]["adapter"] == "nemo_gym"
    assert example["source"]["adapter_version"] == 8
    assert example["source"]["benchmark"] == "workplace_assistant"
    assert example["outcome"]["success"] is True
    for filename in ("train.jsonl", "validation.jsonl"):
        metadata = prepared.manifest["prepared_files"][filename]
        assert len(metadata["sha256"]) == 64
        assert metadata["bytes"] > 0


def test_teacher_prompt_profile_is_materialized_and_training_validated(
    tmp_path: Path,
) -> None:
    prepared = _prepare_fixture_dataset(
        [_source(tmp_path, "teacher")],
        tmp_path / "prepared-teacher-prompt",
        system_prompt_profile="teacher",
    )
    train = _read_jsonl(prepared.train_path)
    validation = _read_jsonl(prepared.validation_path)
    assert prepared.manifest["policy"]["system_prompt_profile"] == "teacher"
    assert prepared.manifest["policy"]["system_prompt_sha256"] == sha256_text(
        DECOMPOSER_SYSTEM_PROMPT
    )
    assert all(
        record["messages"][0]["content"] == DECOMPOSER_SYSTEM_PROMPT
        for record in [*train, *validation]
    )
    runtime = _validate_dataset_system_prompt_profile(
        prepared.manifest,
        train_dataset=Dataset.from_list(train),
        validation_dataset=Dataset.from_list(validation),
        expected_profile="teacher",
    )
    assert runtime == {
        "profile": "teacher",
        "sha256": sha256_text(DECOMPOSER_SYSTEM_PROMPT),
    }
    with pytest.raises(ValueError, match="does not match"):
        _validate_dataset_system_prompt_profile(
            prepared.manifest,
            train_dataset=Dataset.from_list(train),
            validation_dataset=Dataset.from_list(validation),
            expected_profile="student",
        )


def test_prompt_profile_resolver_and_legacy_policy_are_strict() -> None:
    assert (
        resolve_decomposer_system_prompt("student") == DECOMPOSER_STUDENT_SYSTEM_PROMPT
    )
    assert resolve_decomposer_system_prompt("teacher") == DECOMPOSER_SYSTEM_PROMPT
    assert (
        PolicySpec(
            id="legacy", system_prompt="decomposer_default"
        ).resolved_system_prompt_profile
        == "student"
    )
    assert PolicySpec(id="default").resolved_system_prompt_profile == "teacher"
    with pytest.raises(ValueError, match="Unknown Decomposer prompt profile"):
        resolve_decomposer_system_prompt("unknown")
    with pytest.raises(ValidationError, match="mutually exclusive"):
        PolicySpec(
            id="invalid",
            system_prompt="decomposer_default",
            system_prompt_profile="teacher",
        )


def test_prepare_keeps_parallel_calls_as_emitted(tmp_path: Path) -> None:
    rollout = _rollout(0)
    messages = rollout["final_state"]["messages"]
    new_call, new_result = messages[1]["tool_calls"][0], messages[2]
    run_call, run_result = messages[3]["tool_calls"][0], messages[4]
    extra_new = _call("new", "new-extra", agent_type_id="small")
    extra_run = _call(
        "run", "run-extra", agent_id="subagent-extra", prompt="Do the other part."
    )
    refused_wait = _call("wait", "refused-wait")
    messages[1:5] = [
        _ai(
            "Create both.",
            reasoning="Independent parts.",
            tool_calls=[new_call, extra_new],
        ),
        # Results need not follow call order.
        _result("new", "new-extra", {"agent_id": "subagent-extra"}),
        new_result,
        _ai(
            "Run both.",
            reasoning="Start both.",
            tool_calls=[run_call, extra_run, refused_wait],
        ),
        # The harness answers the refused wait before executing the runs.
        _result("wait", "refused-wait", PARALLEL_WAIT_CALL_ERROR),
        run_result,
        _result("run", "run-extra", {"agent_run_id": "subagent-run-extra"}),
    ]

    source = _source(tmp_path, "teacher", [rollout], [_materialized(0)])
    prepared = _prepare_fixture_dataset([source], tmp_path / "prepared")
    records = _read_jsonl(prepared.train_path) + _read_jsonl(prepared.validation_path)
    assert len(records) == 1
    record_messages = records[0]["messages"]
    assert [
        [call["id"] for call in message["tool_calls"]]
        for message in record_messages
        if message["role"] == "assistant"
    ] == [
        ["new-0-0", "new-extra"],
        ["run-0-0", "run-extra", "refused-wait"],
        ["wait-0-0"],
        [],
    ]
    assert [
        message["tool_call_id"]
        for message in record_messages
        if message["role"] == "tool"
    ] == ["new-extra", "new-0-0", "refused-wait", "run-0-0", "run-extra", "wait-0-0"]
    run_turn = next(m for m in record_messages if m["content"] == "Run both.")
    assert run_turn["teacher_reasoning"] == "Start both."
    refusal = next(
        m for m in record_messages if m.get("tool_call_id") == "refused-wait"
    )
    assert refusal["content"] == PARALLEL_WAIT_CALL_ERROR
    assert prepared.manifest["preparation"]["adapter_versions"]["nemo_gym"] == 8
    assert "normalization" not in prepared.manifest


def _chat_call(name: str, call_id: str, **arguments: str) -> dict:
    return {
        "type": "function",
        "id": call_id,
        "function": {"name": name, "arguments": arguments},
    }


def _chat_result(name: str, call_id: str, content: str) -> dict:
    return {"role": "tool", "content": content, "tool_call_id": call_id, "name": name}


def _chat_turn(content: str, *calls: dict) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def test_validator_accepts_parallel_calls_and_requires_every_result() -> None:
    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Task."},
        _chat_turn(
            "Go.",
            _chat_call("new", "new-a", agent_type_id="small"),
            _chat_call("wait", "wait-mixed"),
        ),
        _chat_result("wait", "wait-mixed", PARALLEL_WAIT_CALL_ERROR),
        _chat_result("new", "new-a", '{"agent_id": "a"}'),
        _chat_turn("", _chat_call("wait", "wait-1")),
        _chat_result("wait", "wait-1", "[]"),
        _chat_turn("Done."),
    ]
    validate_decomposer_messages(messages, subagent_type_ids=frozenset({"small"}))

    missing = [*messages[:4], *messages[5:]]
    with pytest.raises(TraceValidationError, match="before tool results"):
        validate_decomposer_messages(missing)


def test_prompt_shared_by_two_categories_stays_on_one_side(tmp_path: Path) -> None:
    # Tasks 0 and 1 carry the same prompt under different categories, like a tau2
    # domain and its `_dsh` implementation.
    rollouts = [_rollout(index, prompt_task_index=0 if index < 2 else None) for index in range(10)]
    materialized = [
        _materialized(index, prompt_task_index=0 if index < 2 else None) for index in range(10)
    ]
    source = _source(tmp_path, "teacher", rollouts, materialized)

    prepared = _prepare_fixture_dataset(
        [source], tmp_path / "prepared", validation_fraction=0.5
    )

    train = _read_jsonl(prepared.train_path)
    validation = _read_jsonl(prepared.validation_path)
    shared = {record["id"] for record in [*train, *validation]} & {
        f"nemo_gym:workplace_assistant:teacher:{index}:0" for index in (0, 1)
    }
    assert len(shared) == 2
    sides = [
        {record["id"] for record in split} & shared for split in (train, validation)
    ]
    assert sorted(len(side) for side in sides) == [0, 2]
    split = prepared.manifest["split"]
    assert split["num_groups"] == 9
    assert split["multi_category_groups"] == 1


def test_prepare_is_reproducible(tmp_path: Path) -> None:
    sources = [_source(tmp_path, "teacher-a"), _source(tmp_path, "teacher-b")]
    first = _prepare_fixture_dataset(
        sources, tmp_path / "first", validation_fraction=0.2
    )
    second = _prepare_fixture_dataset(
        sources, tmp_path / "second", validation_fraction=0.2
    )
    assert first.train_path.read_bytes() == second.train_path.read_bytes()
    assert first.validation_path.read_bytes() == second.validation_path.read_bytes()
    assert (
        first.manifest["split"]["validation_group_ids"]
        == second.manifest["split"]["validation_group_ids"]
    )


def test_dataset_fingerprint_is_portable_across_output_roots(tmp_path: Path) -> None:
    source = _source(tmp_path, "teacher")
    spec = BuildSpec(
        spec_version=1,
        dataset=DatasetIdentity(id="portable", version="v3"),
        policy=PolicySpec(id="decomposer-default"),
        sources=(
            SourceSpec(
                id="source",
                adapter="nemo_gym",
                path=source,
                benchmark="workplace_assistant",
                environment="workplace",
                partition="train",
                teacher="teacher",
            ),
        ),
        selection=SelectionSpec(),
        split=SplitSpec(strategy="prompt_fixed", validation_fraction=0.2, seed=42),
    )
    loaded = LoadedBuildSpec(path=tmp_path / "spec.yaml", sha256="1" * 64, spec=spec)
    first = prepare_dataset(
        loaded,
        tmp_path / "root-a",
        git_revision="test-revision",
        require_clean_git=False,
    )
    second = prepare_dataset(
        loaded,
        tmp_path / "root-b",
        git_revision="test-revision",
        require_clean_git=False,
    )
    assert first.train_path.read_bytes() == second.train_path.read_bytes()
    assert first.validation_path.read_bytes() == second.validation_path.read_bytes()
    assert (
        first.manifest["dataset"]["fingerprint"]
        == second.manifest["dataset"]["fingerprint"]
    )


def test_build_spec_is_strict_and_rejects_test_partitions(tmp_path: Path) -> None:
    raw = {
        "spec_version": 1,
        "dataset": {"id": "strict", "version": "v3"},
        "policy": {"id": "decomposer-default"},
        "sources": [
            {
                "id": "source",
                "adapter": "nemo_gym",
                "path": "source",
                "benchmark": "benchmark",
                "environment": "environment",
                "partition": "test",
                "teacher": "teacher",
            }
        ],
        "selection": {"policy": "exact_reward"},
        "split": {
            "strategy": "prompt_fixed",
            "validation_fraction": 0.1,
            "seed": 42,
        },
        "unknown": True,
    }
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="unknown"):
        load_build_spec(path)

    raw.pop("unknown")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="test partitions"):
        load_build_spec(path)


def test_preserve_split_keeps_declared_source_partitions(tmp_path: Path) -> None:
    train_source = _source(
        tmp_path,
        "teacher-train",
        [_rollout(index) for index in range(5)],
        [_materialized(index) for index in range(5)],
    )
    validation_source = _source(
        tmp_path,
        "teacher-validation",
        [_rollout(index) for index in range(5, 10)],
        [_materialized(index) for index in range(5, 10)],
    )
    spec = BuildSpec(
        spec_version=1,
        dataset=DatasetIdentity(id="preserved", version="v3"),
        policy=PolicySpec(id="decomposer-default"),
        sources=(
            SourceSpec(
                id="train-source",
                adapter="nemo_gym",
                path=train_source,
                benchmark="workplace_assistant",
                environment="workplace",
                partition="train",
                teacher="teacher-train",
            ),
            SourceSpec(
                id="validation-source",
                adapter="nemo_gym",
                path=validation_source,
                benchmark="workplace_assistant",
                environment="workplace",
                partition="validation",
                teacher="teacher-validation",
            ),
        ),
        selection=SelectionSpec(),
        split=SplitSpec(strategy="preserve", seed=42),
    )
    prepared = prepare_dataset(
        LoadedBuildSpec(path=tmp_path / "spec.yaml", sha256="2" * 64, spec=spec),
        tmp_path / "datasets",
        git_revision="test-revision",
        require_clean_git=False,
    )
    assert {
        record["source"]["partition"] for record in _read_jsonl(prepared.train_path)
    } == {"train"}
    assert {
        record["source"]["partition"]
        for record in _read_jsonl(prepared.validation_path)
    } == {"validation"}
    assert prepared.manifest["split"]["strategy"] == "preserve"


def test_training_manifest_validation_checks_prepared_file_hashes(
    tmp_path: Path,
) -> None:
    prepared = _prepare_fixture_dataset(
        [_source(tmp_path, "teacher")],
        tmp_path / "prepared",
        validation_fraction=0.2,
    )
    train = Dataset.from_list(_read_jsonl(prepared.train_path))
    validation = Dataset.from_list(_read_jsonl(prepared.validation_path))
    _validate_manifest(
        prepared.manifest_path,
        prepared.train_path,
        prepared.validation_path,
        train,
        validation,
        limited=False,
    )

    with prepared.train_path.open("a", encoding="utf-8") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="bytes, but the manifest expects"):
        _validate_manifest(
            prepared.manifest_path,
            prepared.train_path,
            prepared.validation_path,
            train,
            validation,
            limited=False,
        )


def _snapshot_spec(snapshot: str, *, version: str = "v1") -> LoadedBuildSpec:
    spec = BuildSpec(
        spec_version=4,
        dataset=DatasetIdentity(id="snapshot-fixture", version=version),
        policy=PolicySpec(id="decomposer-default"),
        sources=(
            SourceSpec(
                id="workplace",
                adapter="nemo_gym",
                snapshot=snapshot,
                benchmark="workplace_assistant",
                environment="workplace",
                partition="train",
                teacher="teacher",
                expected_native_rollouts=11,
                expected_candidates=11,
            ),
        ),
        selection=SelectionSpec(),
        split=SplitSpec(strategy="prompt_fixed", validation_fraction=0.1, seed=42),
    )
    return LoadedBuildSpec(path=Path("spec.yaml"), sha256="0" * 64, spec=spec)


def test_spec_v4_builds_only_from_the_pinned_snapshot(tmp_path: Path) -> None:
    source_dir = _source(tmp_path, "teacher")
    snapshot_dir, snapshot = create_snapshot(
        "nemo_gym",
        source_dir,
        tmp_path / "snapshots",
        SNAPSHOT_FILES["nemo_gym"](source_dir),
    )
    reference = snapshot["digest"]

    prepared = prepare_dataset(
        _snapshot_spec(reference),
        tmp_path / "datasets",
        git_revision="test-revision",
        require_clean_git=False,
        snapshot_root=tmp_path / "snapshots",
    )
    assert prepared.manifest["records"]["total"] == 10
    assert prepared.manifest["sources"][0]["snapshot"] == reference
    assert prepared.manifest["sources"][0]["locator"] == str(snapshot_dir)
    assert prepared.manifest["build_spec"]["config"]["sources"][0]["snapshot"] == reference

    (snapshot_dir / "rollouts.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="was modified"):
        prepare_dataset(
            _snapshot_spec(reference, version="v2"),
            tmp_path / "datasets",
            git_revision="test-revision",
            require_clean_git=False,
            snapshot_root=tmp_path / "snapshots",
        )


def test_spec_v4_requires_snapshots_and_older_specs_reject_them() -> None:
    spec = _snapshot_spec("sha256:" + "a" * 64).spec.model_dump(mode="json")
    with_path = {**spec["sources"][0], "snapshot": None, "path": "/tmp/source"}
    with pytest.raises(ValidationError, match="must name a snapshot"):
        BuildSpec.model_validate({**spec, "sources": [with_path]})
    with pytest.raises(ValidationError, match="require spec_version 4"):
        BuildSpec.model_validate({**spec, "spec_version": 3})
    with pytest.raises(ValidationError, match="sha256:<64 hex>"):
        SourceSpec.model_validate({**spec["sources"][0], "snapshot": "abc"})


def test_training_can_pin_the_expected_release_fingerprint(tmp_path: Path) -> None:
    prepared = _prepare_fixture_dataset(
        [_source(tmp_path, "teacher")],
        tmp_path / "prepared",
        validation_fraction=0.2,
    )
    train = Dataset.from_list(_read_jsonl(prepared.train_path))
    validation = Dataset.from_list(_read_jsonl(prepared.validation_path))
    fingerprint = prepared.manifest["dataset"]["fingerprint"]
    arguments = (
        prepared.manifest_path,
        prepared.train_path,
        prepared.validation_path,
        train,
        validation,
    )
    _validate_manifest(*arguments, limited=False, expected_fingerprint=fingerprint)
    with pytest.raises(ValueError, match="data.expected_fingerprint pins"):
        _validate_manifest(*arguments, limited=False, expected_fingerprint="0" * 64)


def test_training_rejects_legacy_or_tampered_manifests(tmp_path: Path) -> None:
    prepared = _prepare_fixture_dataset(
        [_source(tmp_path, "teacher")],
        tmp_path / "prepared",
        validation_fraction=0.2,
    )
    train = Dataset.from_list(_read_jsonl(prepared.train_path))
    validation = Dataset.from_list(_read_jsonl(prepared.validation_path))
    manifest = json.loads(prepared.manifest_path.read_text())

    manifest["format_version"] = 2
    prepared.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format_version must be 3"):
        _validate_manifest(
            prepared.manifest_path,
            prepared.train_path,
            prepared.validation_path,
            train,
            validation,
            limited=False,
        )

    manifest["format_version"] = 3
    manifest["records"]["train"] += 1
    prepared.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint"):
        _validate_manifest(
            prepared.manifest_path,
            prepared.train_path,
            prepared.validation_path,
            train,
            validation,
            limited=False,
        )


def test_prepare_excludes_malformed_successful_traces_by_reason(tmp_path: Path) -> None:
    rollouts = [_rollout(index) for index in range(9)]
    materialized = [_materialized(index) for index in range(9)]
    rollouts[1]["agent_ref"]["name"] = "simple_agent"
    rollouts[2].pop("final_state")
    rollouts[3]["final_state"]["messages"][-1]["content"] = ""
    rollouts[4]["final_state"]["messages"][1]["tool_calls"][0]["args"] = {}
    rollouts[5]["final_state"]["messages"][1]["tool_calls"].append(
        {"name": "wait", "args": {}, "id": "extra"}
    )
    rollouts[6]["responses_create_params"]["input"] = _input(999)
    rollouts[7]["response"]["tools"][0]["name"] = "other"
    rollouts[8]["reward"] = 0.0
    rollouts.append(_legacy_rollout(9))
    materialized.append(_materialized(9))
    source = _source(tmp_path, "teacher", rollouts, materialized)
    with (source / "rollouts.jsonl").open("a", encoding="utf-8") as file:
        file.write("{not json}\n")

    prepared = _prepare_fixture_dataset([source], tmp_path / "prepared")
    filtering = prepared.manifest["filtering"]
    # rollouts[4]'s argument-less `new` is a mistake the core answers, so it stays.
    assert filtering["included"] == 2
    assert filtering["excluded_invalid_agent_ref"] == 1
    assert filtering["excluded_missing_final_state"] == 1
    assert filtering["excluded_empty_training_target"] == 1
    assert filtering["excluded_invalid_tool_calls"] == 1
    assert filtering["excluded_prompt_mismatch"] == 1
    assert filtering["excluded_invalid_tool_schema"] == 1
    assert filtering["excluded_reward"] == 1
    assert filtering["excluded_invalid_json"] == 1
    assert filtering["excluded_legacy_tool_interface"] == 1


def test_schema_rejects_legacy_spawn_subagent_traces() -> None:
    def reason(function: Callable[[], object]) -> str:
        with pytest.raises(TraceValidationError) as error:
            function()
        return error.value.reason

    legacy_chat_tools = [
        {
            "type": "function",
            "function": {
                key: tool[key] for key in ("name", "description", "parameters")
            },
        }
        for tool in LEGACY_TOOLS
    ]
    spawn = {
        "type": "function",
        "id": "spawn-1",
        "function": {
            "name": "spawn_subagent",
            "arguments": {"subagent_type_id": "small", "prompt": "Do it."},
        },
    }
    legacy_messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Task."},
        {"role": "assistant", "content": "", "tool_calls": [spawn]},
        {
            "role": "tool",
            "content": '{"subagent_run_id": "run"}',
            "tool_call_id": "spawn-1",
            "name": "spawn_subagent",
        },
        {"role": "assistant", "content": "Done.", "tool_calls": []},
    ]
    parallel_legacy = deepcopy(legacy_messages)
    parallel_legacy[2]["tool_calls"].append({**spawn, "id": "spawn-2"})
    parallel_legacy.insert(4, {**legacy_messages[3], "tool_call_id": "spawn-2"})

    legacy = "excluded_legacy_tool_interface"
    assert reason(lambda: normalize_response_tools(deepcopy(LEGACY_TOOLS))) == legacy
    assert reason(lambda: validate_chat_tools(legacy_chat_tools)) == legacy
    assert reason(lambda: validate_decomposer_messages(legacy_messages)) == legacy
    assert reason(lambda: validate_decomposer_messages(parallel_legacy)) == legacy
    # A tool set missing fork/run is structurally invalid, not legacy.
    assert (
        reason(lambda: normalize_response_tools(deepcopy(TOOLS[:1] + TOOLS[3:])))
        == "excluded_invalid_tool_schema"
    )
    assert [
        tool["function"]["name"] for tool in normalize_response_tools(deepcopy(TOOLS))
    ] == ["new", "fork", "run", "wait"]


def test_builder_canonical_tools_are_exactly_new_fork_run_wait(
    tmp_path: Path,
) -> None:
    source = _source(
        tmp_path,
        "teacher",
        [_rollout(index) for index in range(4)],
        [_materialized(index) for index in range(4)],
    )
    subagent_type = {"id": "small", "description": "Fixture subagent."}
    spec = BuildSpec(
        spec_version=2,
        dataset=DatasetIdentity(id="canonical-tools", version="v1"),
        policy=PolicySpec(
            id="decomposer-default",
            system_prompt_profile="teacher",
            subagent_types=(subagent_type,),
        ),
        sources=(
            SourceSpec(
                id="teacher",
                adapter="nemo_gym",
                path=source,
                benchmark="workplace_assistant",
                environment="workplace",
                partition="train",
                teacher="teacher",
                expected_native_rollouts=4,
                expected_candidates=4,
            ),
        ),
        selection=SelectionSpec(),
        split=SplitSpec(strategy="prompt_fixed", validation_fraction=0.25, seed=42),
    )
    prepared = prepare_dataset(
        LoadedBuildSpec(path=tmp_path / "spec.yaml", sha256="4" * 64, spec=spec),
        tmp_path / "datasets",
        git_revision="test-revision",
        require_clean_git=False,
    )
    expected = build_decomposer_chat_tools(
        [
            {
                "agent_type_id": "small",
                "description": "Fixture subagent.",
                "assistant_id": "small",
            }
        ]
    )
    assert [tool["function"]["name"] for tool in expected] == [
        "new",
        "fork",
        "run",
        "wait",
    ]
    records = _read_jsonl(prepared.train_path) + _read_jsonl(prepared.validation_path)
    assert len(records) == 4
    assert all(record["tools"] == expected for record in records)
    assert all(
        record["messages"][0]["content"] == DECOMPOSER_SYSTEM_PROMPT
        for record in records
    )
    assert prepared.manifest["sources"][0]["tool_schema_origin"] == (
        "canonical_policy_interface"
    )
    assert prepared.manifest["content"]["tool_schema_sha256s"] == [
        sha256_text(canonical_json(expected))
    ]


def _renamed_core_rollout(task_index: int) -> dict:
    """new -> run -> wait -> fork -> run -> wait -> answer, as the renamed core records it."""
    rollout = _rollout(task_index)
    state = rollout["final_state"]
    messages = state["messages"]
    messages[1]["tool_calls"][0]["args"]["agent_type_id"] = "small-alias"
    agent_id, fork_id = f"subagent-{task_index}", f"fork-{task_index}"
    messages[-1:-1] = [
        _ai("", reasoning="Fork it.", tool_calls=[_call("fork", f"f-{task_index}", agent_id=agent_id)]),
        _result("fork", f"f-{task_index}", {"agent_id": fork_id}),
        _ai(
            "",
            reasoning="Check it.",
            tool_calls=[_call("run", f"r2-{task_index}", agent_id=fork_id, prompt="Check it.")],
        ),
        _result("run", f"r2-{task_index}", {"agent_run_id": f"fork-run-{task_index}"}),
        _ai("", reasoning="Wait again.", tool_calls=[_call("wait", f"w2-{task_index}")]),
        _result(
            "wait",
            f"w2-{task_index}",
            [
                {
                    "agent_id": fork_id,
                    "agent_run_id": f"fork-run-{task_index}",
                    "status": "responded",
                    "response": "Checked.",
                    "error": None,
                }
            ],
        ),
    ]
    agent = {"agent_type_id": "small", "assistant_id": "small", "created_at": 0.0}
    state["agents"] = {
        agent_id: {"agent_id": agent_id, "thread_id": agent_id, **agent},
        fork_id: {"agent_id": fork_id, "thread_id": fork_id, "forked_from": agent_id, **agent},
    }
    state["agent_runs"] = {
        run_id: {"agent_run_id": run_id, "agent_id": owner, "status": "responded", "collected_at": 1.0}
        for run_id, owner in (
            (f"subagent-run-{task_index}", agent_id),
            (f"fork-run-{task_index}", fork_id),
        )
    }
    state["decomposer_agent_runs"] = []
    return rollout


def test_renamed_core_rollouts_load_with_canonical_tools(tmp_path: Path) -> None:
    source = _source(
        tmp_path,
        "teacher",
        [_renamed_core_rollout(index) for index in range(4)],
        [_materialized(index) for index in range(4)],
    )
    spec = BuildSpec(
        spec_version=2,
        dataset=DatasetIdentity(id="renamed-core", version="v1"),
        policy=PolicySpec(
            id="decomposer-default",
            system_prompt_profile="teacher",
            subagent_types=({"id": "small", "description": "Fixture subagent."},),
        ),
        sources=(
            SourceSpec(
                id="teacher",
                adapter="nemo_gym",
                path=source,
                benchmark="workplace_assistant",
                environment="workplace",
                partition="train",
                teacher="teacher",
                expected_native_rollouts=4,
                expected_candidates=4,
                subagent_type_aliases={"small-alias": "small"},
            ),
        ),
        selection=SelectionSpec(),
        split=SplitSpec(strategy="prompt_fixed", validation_fraction=0.25, seed=42),
    )
    prepared = prepare_dataset(
        LoadedBuildSpec(path=tmp_path / "spec.yaml", sha256="5" * 64, spec=spec),
        tmp_path / "datasets",
        git_revision="test-revision",
        require_clean_git=False,
    )
    expected_tools = build_decomposer_chat_tools(
        [{"agent_type_id": "small", "description": "Fixture subagent.", "assistant_id": "small"}]
    )
    records = _read_jsonl(prepared.train_path) + _read_jsonl(prepared.validation_path)
    assert len(records) == 4
    for record in records:
        assert record["tools"] == expected_tools
        calls = [
            call["function"]
            for message in record["messages"]
            for call in message.get("tool_calls") or []
        ]
        assert [call["name"] for call in calls] == ["new", "run", "wait", "fork", "run", "wait"]
        assert calls[0]["arguments"] == {"agent_type_id": "small"}
        assert set(calls[3]["arguments"]) == {"agent_id"}
        assert set(calls[4]["arguments"]) == {"agent_id", "prompt"}


def _core_answered_mistakes_rollout(task_index: int) -> dict:
    """Mistakes the core answers, each followed by the manager's recovery."""
    rollout = _rollout(task_index)
    agent_id, agent_run_id = f"subagent-{task_index}", f"subagent-run-{task_index}"
    rollout["final_state"]["messages"] = [
        rollout["final_state"]["messages"][0],
        _ai(
            "",
            tool_calls=[
                _call("list_claims", "lc"),
                _call("new", "quoted", agent_type_id='"small"'),
            ],
        ),
        _result(
            "list_claims",
            "lc",
            "Error: list_claims is not a valid tool, try one of [new, fork, run, wait].",
        ),
        _result("new", "quoted", 'Unknown agent type ID `"small"`. Available IDs: `small`.'),
        _ai("", tool_calls=[_call("new", "new", agent_type_id="small")]),
        _result("new", "new", {"agent_id": agent_id}),
        _ai(
            "",
            tool_calls=[
                _call("run", "run", agent_id=agent_id, prompt="Do it.", agent_id_note="x")
            ],
        ),
        _result("run", "run", {"agent_run_id": agent_run_id}),
        _ai("4."),
        {"type": "human", "content": EARLY_RESPONSE_ERROR},
        _ai("", tool_calls=[_call("wait", "wait", agent_id=agent_id)]),
        _result(
            "wait",
            "wait",
            [{"agent_id": agent_id, "agent_run_id": agent_run_id, "status": "responded", "response": "4"}],
        ),
        _ai("The answer is 4."),
    ]
    return rollout


def test_nemo_gym_keeps_mistakes_the_core_answered(tmp_path: Path) -> None:
    other_user_message = _core_answered_mistakes_rollout(1)
    other_user_message["final_state"]["messages"][9]["content"] = "Please hurry."
    source = _source(
        tmp_path,
        "teacher",
        [_core_answered_mistakes_rollout(0), other_user_message],
        [_materialized(0), _materialized(1)],
    )

    prepared = _prepare_fixture_dataset([source], tmp_path / "prepared")

    filtering = prepared.manifest["filtering"]
    assert filtering["included"] == 1
    assert filtering["excluded_invalid_messages"] == 1
    [record] = _read_jsonl(prepared.train_path) + _read_jsonl(prepared.validation_path)
    messages = record["messages"]
    assert [message["role"] for message in messages] == [
        "system", "user",
        "assistant", "tool", "tool",
        "assistant", "tool",
        "assistant", "tool",
        "assistant", "user",
        "assistant", "tool",
        "assistant",
    ]
    calls = [
        call["function"]
        for message in messages
        for call in message.get("tool_calls") or []
    ]
    assert [call["name"] for call in calls] == ["list_claims", "new", "new", "run", "wait"]
    assert calls[1]["arguments"] == {"agent_type_id": '"small"'}
    assert calls[3]["arguments"]["agent_id_note"] == "x"
    assert calls[4]["arguments"] == {"agent_id": "subagent-0"}
    assert messages[10]["content"] == EARLY_RESPONSE_ERROR


def test_v2_without_subagent_types_keeps_each_source_native_schema(
    tmp_path: Path,
) -> None:
    sources = []
    for name, description in (("gym-a", "The new tool."), ("gym-b", "Other types.")):
        rollouts = [_rollout(index) for index in range(4)]
        for rollout in rollouts:
            rollout["response"]["tools"][0]["description"] = description
        sources.append(
            SourceSpec(
                id=name,
                adapter="nemo_gym",
                path=_source(
                    tmp_path, name, rollouts, [_materialized(index) for index in range(4)]
                ),
                benchmark=name,
                environment="workplace",
                partition="train",
                teacher="teacher",
                expected_native_rollouts=4,
                expected_candidates=4,
            )
        )
    spec = BuildSpec(
        spec_version=2,
        dataset=DatasetIdentity(id="native-schemas", version="v1"),
        policy=PolicySpec(id="decomposer-default"),
        sources=tuple(sources),
        selection=SelectionSpec(),
        split=SplitSpec(strategy="prompt_fixed", validation_fraction=0.25, seed=42),
    )

    prepared = prepare_dataset(
        LoadedBuildSpec(path=tmp_path / "spec.yaml", sha256="6" * 64, spec=spec),
        tmp_path / "datasets",
        git_revision="test-revision",
        require_clean_git=False,
    )

    records = _read_jsonl(prepared.train_path) + _read_jsonl(prepared.validation_path)
    assert len(records) == 8
    hashes = {}
    for source_manifest in prepared.manifest["sources"]:
        assert source_manifest["tool_schema_origin"] == "response.tools"
        hashes[source_manifest["id"]] = source_manifest["tool_schema_sha256"]
    assert hashes["gym-a"] != hashes["gym-b"]
    assert prepared.manifest["content"]["tool_schema_sha256s"] == sorted(hashes.values())
    for record in records:
        assert sha256_text(canonical_json(record["tools"])) == hashes[
            record["source"]["source_id"]
        ]
        assert record["messages"][0]["content"] == DECOMPOSER_SYSTEM_PROMPT
    assert prepared.manifest["policy"]["system_prompt_profile"] == "teacher"


def test_invalid_policy_error_reports_source_line(tmp_path: Path) -> None:
    rollouts = [_rollout(0), _rollout(1)]
    rollouts[1]["final_state"]["messages"][2]["tool_call_id"] = "unknown"
    source = _source(
        tmp_path,
        "teacher",
        rollouts,
        [_materialized(0), _materialized(1)],
    )
    with pytest.raises(ValueError, match=r"rollouts\.jsonl:2:.*unknown or mismatched"):
        _prepare_fixture_dataset(
            [source], tmp_path / "prepared", invalid_policy="error"
        )


@pytest.mark.parametrize(
    ("reason", "mutate_rollout", "mutate_materialized"),
    [
        (
            "excluded_invalid_reward",
            lambda rollout: rollout.update({"reward": "1.0"}),
            lambda materialized: None,
        ),
        (
            "excluded_invalid_indices",
            lambda rollout: rollout.update({"_ng_rollout_index": True}),
            lambda materialized: None,
        ),
        (
            "excluded_missing_materialized_input",
            lambda rollout: None,
            lambda materialized: materialized.clear(),
        ),
        (
            "excluded_invalid_messages",
            lambda rollout: rollout["final_state"]["messages"][0].update(
                {"type": "tool"}
            ),
            lambda materialized: None,
        ),
        (
            "excluded_invalid_metadata",
            lambda rollout: None,
            lambda materialized: materialized[0].update({"category": ""}),
        ),
    ],
)
def test_additional_row_exclusion_reasons(
    tmp_path: Path,
    reason: str,
    mutate_rollout: Callable[[dict], None],
    mutate_materialized: Callable[[list[dict]], None],
) -> None:
    valid_rollout = _rollout(0)
    invalid_rollout = _rollout(1)
    invalid_materialized = [_materialized(1)]
    mutate_rollout(invalid_rollout)
    mutate_materialized(invalid_materialized)
    source = _source(
        tmp_path,
        "teacher",
        [valid_rollout, invalid_rollout],
        [_materialized(0), *invalid_materialized],
    )
    prepared = _prepare_fixture_dataset([source], tmp_path / "prepared")
    assert prepared.manifest["filtering"]["included"] == 1
    assert prepared.manifest["filtering"][reason] == 1


def test_prompt_teacher_cap_is_deterministic_and_keeps_split_groups(
    tmp_path: Path,
) -> None:
    sources = []
    for teacher in ("teacher-a", "teacher-b"):
        rollouts = [
            _rollout(0, rollout_index=index, prompt_task_index=0) for index in range(5)
        ] + [
            _rollout(1, rollout_index=index, prompt_task_index=1) for index in range(5)
        ]
        materialized = [
            _materialized(0, rollout_index=index, prompt_task_index=0)
            for index in range(5)
        ] + [
            _materialized(1, rollout_index=index, prompt_task_index=1)
            for index in range(5)
        ]
        sources.append(_source(tmp_path, teacher, rollouts, materialized))

    first = _prepare_fixture_dataset(
        sources,
        tmp_path / "first",
        validation_fraction=0.5,
        max_traces_per_prompt_per_teacher=2,
    )
    second = _prepare_fixture_dataset(
        sources,
        tmp_path / "second",
        validation_fraction=0.5,
        max_traces_per_prompt_per_teacher=2,
    )
    assert first.manifest["filtering"]["eligible_before_cap"] == 20
    assert first.manifest["filtering"]["excluded_prompt_teacher_cap"] == 12
    assert first.manifest["filtering"]["included"] == 8
    assert first.train_path.read_bytes() == second.train_path.read_bytes()
    assert first.validation_path.read_bytes() == second.validation_path.read_bytes()
    train = _read_jsonl(first.train_path)
    validation = _read_jsonl(first.validation_path)
    assert {record["group_id"] for record in train}.isdisjoint(
        record["group_id"] for record in validation
    )


def test_v2_samples_before_validation_and_keeps_all_rewards(tmp_path: Path) -> None:
    source_id = "teacher"
    rollouts = [
        _rollout(
            task_index,
            rollout_index=rollout_index,
            reward=0.0 if task_index == 0 else 1.0,
        )
        for task_index in range(3)
        for rollout_index in range(3)
    ]
    materialized = [
        _materialized(task_index, rollout_index=rollout_index)
        for task_index in range(3)
        for rollout_index in range(3)
    ]

    def selected_rollout(task_index: int) -> int:
        return min(
            range(3),
            key=lambda rollout_index: sha256_text(
                "42\0nemo_gym:workplace_assistant:"
                f"{source_id}:{task_index}:{rollout_index}"
            ),
        )

    selected_invalid = next(
        rollout
        for rollout in rollouts
        if rollout["_ng_task_index"] == 1
        and rollout["_ng_rollout_index"] == selected_rollout(1)
    )
    selected_invalid["final_state"]["messages"][1]["tool_calls"][0]["args"][
        "agent_type_id"
    ] = "unknown"
    unselected_invalid = next(
        rollout
        for rollout in rollouts
        if rollout["_ng_task_index"] == 2
        and rollout["_ng_rollout_index"] != selected_rollout(2)
    )
    unselected_invalid["final_state"] = None

    source = _source(tmp_path, source_id, rollouts, materialized)
    spec = BuildSpec(
        spec_version=2,
        dataset=DatasetIdentity(id="sample-before-filter", version="v1"),
        policy=PolicySpec(
            id="decomposer-default",
            subagent_types=(
                {
                    "id": "small",
                    "description": "General-purpose fixture subagent.",
                },
            ),
        ),
        sources=(
            SourceSpec(
                id=source_id,
                adapter="nemo_gym",
                path=source,
                benchmark="workplace_assistant",
                environment="workplace",
                partition="train",
                teacher="teacher",
                sampling={
                    "strategy": "task_hash",
                    "seed": 42,
                    "max_per_task": 1,
                    "expected_tasks": 3,
                    "expected_rollouts_per_task": 3,
                },
                expected_native_rollouts=9,
                expected_candidates=3,
            ),
        ),
        selection=SelectionSpec(policy="all_rewards"),
        split=SplitSpec(strategy="prompt_fixed", validation_fraction=0.5, seed=42),
    )
    prepared = prepare_dataset(
        LoadedBuildSpec(path=tmp_path / "spec.yaml", sha256="2" * 64, spec=spec),
        tmp_path / "datasets",
        git_revision="test-revision",
        require_clean_git=False,
    )
    records = _read_jsonl(prepared.train_path) + _read_jsonl(prepared.validation_path)
    assert len(records) == 2
    assert {record["outcome"]["success"] for record in records} == {False, True}
    assert {record["outcome"]["reward"] for record in records} == {0.0, 1.0}
    assert all(record["tools"] == records[0]["tools"] for record in records)
    filtering = prepared.manifest["filtering"]
    assert filtering["rollouts"] == 3
    assert filtering["excluded_reward"] == 0
    assert filtering["excluded_malformed"] == 1
    assert filtering["excluded_malformed_by_source"] == {source_id: 1}
    assert filtering["excluded_invalid_tool_calls"] == 1
    assert filtering["included"] == 2
    source_manifest = prepared.manifest["sources"][0]
    assert source_manifest["native_rollouts"] == 9
    assert source_manifest["candidate_rollouts"] == 3
    assert source_manifest["sampling"]["not_selected"] == 6


def test_prepare_refuses_overwrite_and_bad_materialized_source(tmp_path: Path) -> None:
    source = _source(tmp_path, "teacher")
    output = tmp_path / "prepared"
    _prepare_fixture_dataset([source], output)
    with pytest.raises(FileExistsError):
        _prepare_fixture_dataset([source], output)

    bad_source = _source(tmp_path, "bad-teacher")
    materialized = _read_jsonl(bad_source / "rollouts_materialized_inputs.jsonl")
    materialized.append(materialized[0])
    _write_jsonl(bad_source / "rollouts_materialized_inputs.jsonl", materialized)
    with pytest.raises(ValueError, match="Duplicate materialized input"):
        _prepare_fixture_dataset([bad_source], tmp_path / "bad-output")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda rollout: rollout["final_state"]["messages"][2].update(
                {"tool_call_id": "unknown"}
            ),
            "unknown or mismatched",
        ),
        (
            lambda rollout: rollout["final_state"]["messages"][3]["tool_calls"][
                0
            ].update(
                {"id": rollout["final_state"]["messages"][1]["tool_calls"][0]["id"]}
            ),
            "Duplicate tool-call ID",
        ),
        (
            lambda rollout: rollout["final_state"]["messages"][1]["tool_calls"].append(
                _call(
                    "run",
                    "missing-result",
                    agent_id="subagent-0",
                    prompt="Missing result.",
                )
            ),
            "before tool results",
        ),
        (
            lambda rollout: rollout.update(_legacy_rollout(0)),
            "retired 'spawn_subagent' tool",
        ),
    ],
)
def test_strict_tool_call_validation_in_error_mode(
    tmp_path: Path, mutation: Callable[[dict], None], match: str
) -> None:
    rollout = _rollout(0)
    mutation(rollout)
    source = _source(tmp_path, "teacher", [rollout], [_materialized(0)])
    with pytest.raises(ValueError, match=match):
        _prepare_fixture_dataset(
            [source], tmp_path / "prepared", invalid_policy="error"
        )


# Specs that still pin a tokenizer predate tokenizer-free releases and load only
# at the commits that built them.
SFT_SPEC_PATHS = sorted(
    path
    for path in Path(__file__).resolve().parents[1].glob("sft/**/specs/*.yaml")
    if "tokenization" not in yaml.safe_load(path.read_text())
)


def test_sft_specs_live_under_sft():
    assert SFT_SPEC_PATHS


@pytest.mark.parametrize("spec_path", SFT_SPEC_PATHS, ids=lambda path: path.name)
def test_every_sft_spec_loads_and_its_manifests_exist(spec_path):
    loaded = load_build_spec(spec_path)

    if loaded.spec.split.manifest is not None:
        assert loaded.spec.split.manifest.is_file()
    for source in loaded.spec.sources:
        if source.gaia2 is not None:
            assert source.gaia2.split_manifest.is_file()
