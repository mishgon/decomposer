from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from datasets import Dataset
from pydantic import ValidationError

from data.sft.builder import LoadedBuildSpec, load_build_spec, prepare_dataset
from data.sft.schema import (
    EXCLUSION_REASONS,
    BuildSpec,
    DatasetIdentity,
    PolicySpec,
    SelectionSpec,
    SourceSpec,
    SplitSpec,
)
from training.sft.train import _validate_manifest

TOOLS = [
    {
        "name": "spawn_subagent",
        "description": "Spawn one subagent.",
        "parameters": {
            "type": "object",
            "properties": {
                "subagent_type_id": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["subagent_type_id", "prompt"],
        },
        "strict": False,
        "type": "function",
    },
    {
        "name": "wait",
        "description": "Wait for reports.",
        "parameters": {"type": "object", "properties": {}},
        "strict": False,
        "type": "function",
    },
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


def _rollout(
    task_index: int,
    *,
    rollout_index: int = 0,
    reward: float = 1.0,
    prompt_task_index: int | None = None,
) -> dict:
    spawn_id = f"spawn-{task_index}-{rollout_index}"
    wait_id = f"wait-{task_index}-{rollout_index}"
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
                    tool_calls=[
                        {
                            "name": "spawn_subagent",
                            "args": {
                                "subagent_type_id": "small",
                                "prompt": f"Do task {task_index}.",
                            },
                            "id": spawn_id,
                        }
                    ],
                ),
                {
                    "type": "tool",
                    "content": json.dumps({"subagent_run_id": f"run-{task_index}"}),
                    "tool_call_id": spawn_id,
                    "name": "spawn_subagent",
                },
                _ai(
                    "",
                    reasoning="Wait.",
                    tool_calls=[{"name": "wait", "args": {}, "id": wait_id}],
                ),
                {
                    "type": "tool",
                    "content": json.dumps(
                        [
                            {
                                "subagent_run_id": f"run-{task_index}",
                                "status": "success",
                                "content": "Done.",
                            }
                        ]
                    ),
                    "tool_call_id": wait_id,
                    "name": "wait",
                },
                _ai("The task is complete.", reasoning="Report success."),
            ]
        },
        "_ng_task_index": task_index,
        "_ng_rollout_index": rollout_index,
    }


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
):
    """Test helper that exercises the new canonical builder without Git state."""
    spec = BuildSpec(
        spec_version=1,
        dataset=DatasetIdentity(id=output_dir.name, version="v3"),
        policy=PolicySpec(id="decomposer-default"),
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
    assert example["messages"][1]["role"] == "user"
    assert example["messages"][-1]["teacher_reasoning"] == "Report success."
    assert example["tools"][0]["function"]["name"] == "spawn_subagent"
    assert example["source"]["adapter"] == "nemo_gym"
    assert example["source"]["benchmark"] == "workplace_assistant"
    assert example["outcome"]["success"] is True
    for filename in ("train.jsonl", "validation.jsonl"):
        metadata = prepared.manifest["prepared_files"][filename]
        assert len(metadata["sha256"]) == 64
        assert metadata["bytes"] > 0


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
    source = _source(tmp_path, "teacher", rollouts, materialized)
    with (source / "rollouts.jsonl").open("a", encoding="utf-8") as file:
        file.write("{not json}\n")

    prepared = _prepare_fixture_dataset([source], tmp_path / "prepared")
    filtering = prepared.manifest["filtering"]
    assert filtering["included"] == 1
    assert filtering["excluded_invalid_agent_ref"] == 1
    assert filtering["excluded_missing_final_state"] == 1
    assert filtering["excluded_empty_training_target"] == 1
    assert filtering["excluded_invalid_tool_calls"] == 1
    assert filtering["excluded_multiple_tool_calls"] == 1
    assert filtering["excluded_prompt_mismatch"] == 1
    assert filtering["excluded_invalid_tool_schema"] == 1
    assert filtering["excluded_reward"] == 1
    assert filtering["excluded_invalid_json"] == 1


def test_invalid_policy_error_reports_source_line(tmp_path: Path) -> None:
    rollouts = [_rollout(0), _rollout(1)]
    rollouts[1]["final_state"]["messages"][1]["tool_calls"][0]["args"] = {}
    source = _source(
        tmp_path,
        "teacher",
        rollouts,
        [_materialized(0), _materialized(1)],
    )
    with pytest.raises(ValueError, match=r"rollouts\.jsonl:2:.*subagent_type_id"):
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
            lambda rollout: rollout["final_state"]["messages"][1]["tool_calls"][
                0
            ].update({"name": "spinvoke"}),
            "invalid name or arguments",
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
