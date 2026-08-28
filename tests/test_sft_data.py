from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from datasets import Dataset
from pydantic import ValidationError

from data.sft import builder as builder_module
from data.sft.builder import LoadedBuildSpec, load_build_spec, prepare_dataset
from data.sft.schema import (
    EXCLUSION_REASONS,
    BuildSpec,
    DatasetIdentity,
    PolicySpec,
    SelectionSpec,
    SourceSpec,
    SplitSpec,
    TokenizationSpec,
    sha256_text,
)
from decomposer.prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    DECOMPOSER_TEACHER_SYSTEM_PROMPT,
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
    version: str = "v3",
    tokenization: TokenizationSpec | None = None,
):
    """Test helper that exercises the new canonical builder without Git state."""
    spec = BuildSpec(
        spec_version=1,
        dataset=DatasetIdentity(id=output_dir.name, version=version),
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
        tokenization=tokenization,
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
    assert example["messages"][0]["content"] != DECOMPOSER_TEACHER_SYSTEM_PROMPT
    assert example["messages"][1]["role"] == "user"
    assert example["messages"][-1]["teacher_reasoning"] == "Report success."
    assert example["tools"][0]["function"]["name"] == "spawn_subagent"
    assert example["source"]["adapter"] == "nemo_gym"
    assert example["source"]["adapter_version"] == 3
    assert example["source"]["benchmark"] == "workplace_assistant"
    assert example["outcome"]["success"] is True
    for filename in ("train.jsonl", "validation.jsonl"):
        metadata = prepared.manifest["prepared_files"][filename]
        assert len(metadata["sha256"]) == 64
        assert metadata["bytes"] > 0


class _LengthFixtureTokenizer:
    init_kwargs = {"_commit_hash": "fixture-revision"}

    def apply_chat_template(self, messages, **kwargs):
        prompt = next(
            message["content"] for message in messages if message["role"] == "user"
        )
        token_length = 12 if prompt.endswith("Task 0.") else 6
        return {
            "input_ids": list(range(token_length)),
            "assistant_masks": [1] * token_length,
        }


def test_versioned_token_limits_produce_stable_strict_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_runtime(spec: TokenizationSpec):
        return (
            _LengthFixtureTokenizer(),
            "fixture-template",
            {
                "profile": spec.profile,
                "tokenizer": spec.tokenizer,
                "requested_revision": spec.revision,
                "resolved_revision": "fixture-revision",
                "tokenizer_class": "_LengthFixtureTokenizer",
                "canonical_template_sha256": "1" * 64,
                "training_template_sha256": "2" * 64,
                "include_reasoning": False,
                "max_tokens": spec.max_tokens,
            },
        )

    monkeypatch.setattr(builder_module, "_load_tokenization_runtime", fake_runtime)
    source = _source(tmp_path, "teacher")
    common = {
        "profile": "gemma4_sft_non_thinking",
        "tokenizer": "google/gemma-4-E4B-it",
        "revision": "main",
    }
    prepared_8k = _prepare_fixture_dataset(
        [source],
        tmp_path / "prepared-8k",
        version="v2-8k",
        tokenization=TokenizationSpec(**common, max_tokens=8),
    )
    prepared_32k = _prepare_fixture_dataset(
        [source],
        tmp_path / "prepared-32k",
        version="v2-32k",
        tokenization=TokenizationSpec(**common, max_tokens=32),
    )

    rows_8k = {
        row["id"]: split
        for split, path in (
            ("train", prepared_8k.train_path),
            ("validation", prepared_8k.validation_path),
        )
        for row in _read_jsonl(path)
    }
    rows_32k = {
        row["id"]: (split, row)
        for split, path in (
            ("train", prepared_32k.train_path),
            ("validation", prepared_32k.validation_path),
        )
        for row in _read_jsonl(path)
    }
    assert set(rows_8k) < set(rows_32k)
    assert all(rows_32k[row_id][0] == split for row_id, split in rows_8k.items())
    assert prepared_8k.manifest["filtering"]["excluded_token_length"] == 1
    assert prepared_8k.manifest["filtering"]["excluded_token_length_by_source"] == {
        "teacher": 1
    }
    assert prepared_8k.manifest["sources"][0]["tokenization"] == {
        "eligible_before_token_limit": 10,
        "excluded_token_length": 1,
        "included": 9,
    }
    assert prepared_32k.manifest["filtering"]["excluded_token_length"] == 0
    for _, row in rows_32k.values():
        metadata = row["attributes"]["prepared_tokenization"]
        assert metadata["profile"] == "gemma4_sft_non_thinking"
        assert metadata["tokens"] in {6, 12}


@pytest.mark.parametrize("call_count", [2, 3, 7])
def test_prepare_sequentializes_parallel_spawn_calls(
    tmp_path: Path, call_count: int
) -> None:
    rollout = _rollout(0)
    messages = rollout["final_state"]["messages"]
    first_call = messages[1]["tool_calls"][0]
    calls = [
        first_call,
        *[
            {
                "name": "spawn_subagent",
                "args": {
                    "subagent_type_id": "small",
                    "prompt": f"Do independent subtask {index}.",
                },
                "id": f"spawn-{index}",
            }
            for index in range(2, call_count + 1)
        ],
    ]
    messages[1] = _ai(
        "Delegate these in parallel.",
        reasoning="These subtasks are independent.",
        tool_calls=calls,
    )
    first_result = messages[2]
    results = [
        first_result,
        *[
            {
                "type": "tool",
                "content": json.dumps({"subagent_run_id": f"run-{index}"}),
                "tool_call_id": f"spawn-{index}",
                "name": "spawn_subagent",
            }
            for index in range(2, call_count + 1)
        ],
    ]
    # Exercise ID-based matching: native results need not use call order.
    messages[2:3] = list(reversed(results))

    source = _source(tmp_path, "teacher", [rollout], [_materialized(0)])
    prepared = _prepare_fixture_dataset([source], tmp_path / "prepared")
    records = _read_jsonl(prepared.train_path) + _read_jsonl(prepared.validation_path)
    assert len(records) == 1
    record = records[0]
    spawn_assistant_indices = [
        index
        for index, message in enumerate(record["messages"])
        if message["role"] == "assistant"
        and message.get("tool_calls")
        and message["tool_calls"][0]["function"]["name"] == "spawn_subagent"
    ]
    assert len(spawn_assistant_indices) == call_count
    spawn_messages = [record["messages"][index] for index in spawn_assistant_indices]
    expected_ids = [call["id"] for call in calls]
    assert [
        message["tool_calls"][0]["id"] for message in spawn_messages
    ] == expected_ids
    assert [
        record["messages"][index + 1]["tool_call_id"]
        for index in spawn_assistant_indices
    ] == expected_ids
    assert spawn_messages[0]["content"] == "Delegate these in parallel."
    assert spawn_messages[0]["teacher_reasoning"] == ("These subtasks are independent.")
    assert [message["content"] for message in spawn_messages[1:]] == [""] * (
        call_count - 1
    )
    assert all("teacher_reasoning" not in message for message in spawn_messages[1:])
    assert record["attributes"]["parallel_spawn_normalization"] == {
        "messages": 1,
        "tool_calls": call_count,
    }

    assert prepared.manifest["preparation"]["adapter_versions"]["nemo_gym"] == 3
    assert prepared.manifest["normalization"] == {
        "strategy": "parallel_spawn_calls_to_single_call_turns",
        "traces": 1,
        "messages": 1,
        "tool_calls": call_count,
    }
    assert prepared.manifest["sources"][0]["normalization"] == {
        "traces": 1,
        "messages": 1,
        "tool_calls": call_count,
    }
    assert prepared.manifest["filtering"]["included"] == 1
    assert prepared.manifest["filtering"]["excluded_multiple_tool_calls"] == 0


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
    assert filtering["excluded_invalid_tool_calls"] == 2
    assert filtering["excluded_multiple_tool_calls"] == 0
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
        "subagent_type_id"
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
            lambda rollout: rollout["final_state"]["messages"][1]["tool_calls"][
                0
            ].update({"name": "spinvoke"}),
            "invalid name or arguments",
        ),
        (
            lambda rollout: rollout["final_state"]["messages"][1]["tool_calls"].append(
                {
                    "name": "spawn_subagent",
                    "args": {
                        "subagent_type_id": "small",
                        "prompt": "Missing result.",
                    },
                    "id": "missing-result",
                }
            ),
            "exactly one matching tool result",
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


def test_qwen35_workplace_partial_spec_is_pinned_and_success_only() -> None:
    loaded = load_build_spec(
        Path(
            "data/sft/specs/"
            "decomposer_workplace_deepseek_qwen35_4b_nonthinking_"
            "v1_1444_32k.yaml"
        )
    )
    spec = loaded.spec
    assert spec.dataset.id == ("decomposer-workplace-deepseek-qwen35-4b-nonthinking")
    assert spec.dataset.version == "v1-1444-32k"
    assert len(spec.sources) == 1
    source = spec.sources[0]
    assert source.adapter == "nemo_gym"
    assert source.partition == "train"
    assert source.path is not None
    assert source.path.name.endswith("first-1444")
    assert spec.selection.success_reward == 1.0
    assert spec.selection.invalid_policy == "exclude"
    assert spec.selection.max_traces_per_prompt_per_teacher is None
    assert spec.split.strategy == "prompt_fixed"
    assert spec.split.validation_fraction == 0.1
    assert spec.split.seed == 42
    assert spec.tokenization is not None
    assert spec.tokenization.profile == "qwen35_sft_non_thinking"
    assert spec.tokenization.revision == ("851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    assert spec.tokenization.max_tokens == 32768


def test_qwen35_workplace_full_spec_is_pinned_and_success_only() -> None:
    loaded = load_build_spec(
        Path(
            "data/sft/specs/"
            "decomposer_workplace_deepseek_qwen35_4b_nonthinking_"
            "v1_3765_32k.yaml"
        )
    )
    spec = loaded.spec
    assert spec.dataset.id == ("decomposer-workplace-deepseek-qwen35-4b-nonthinking")
    assert spec.dataset.version == "v1-3765-32k"
    assert len(spec.sources) == 1
    source = spec.sources[0]
    assert source.adapter == "nemo_gym"
    assert source.partition == "train"
    assert source.path is not None
    assert source.path.name == ("deepseek-v4-flash-0731-qwen35-4b-non-thinking-n3")
    assert spec.selection.success_reward == 1.0
    assert spec.selection.invalid_policy == "exclude"
    assert spec.selection.max_traces_per_prompt_per_teacher is None
    assert spec.split.strategy == "prompt_fixed"
    assert spec.split.validation_fraction == 0.1
    assert spec.split.seed == 42
    assert spec.tokenization is not None
    assert spec.tokenization.profile == "qwen35_sft_non_thinking"
    assert spec.tokenization.revision == ("851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    assert spec.tokenization.max_tokens == 32768


def test_qwen35_mixed_spec_pins_all_reward_sources_and_sampling() -> None:
    spec = load_build_spec(
        Path(
            "data/sft/specs/decomposer_mixed_deepseek_qwen35_4b_nonthinking_v1_32k.yaml"
        )
    ).spec
    assert spec.spec_version == 2
    assert spec.dataset.version == "v1-final-493c24c4-404-32k"
    assert spec.selection.policy == "all_rewards"
    assert spec.selection.invalid_policy == "exclude"
    assert spec.policy.subagent_types[0].id == "qwen35_4b_non_thinking"
    workplace, toolathlon = spec.sources
    assert workplace.expected_native_rollouts == 3765
    assert workplace.expected_candidates == 1255
    assert workplace.sampling is not None
    assert workplace.sampling.seed == 42
    assert workplace.sampling.expected_tasks == 1255
    assert workplace.sampling.expected_rollouts_per_task == 3
    assert toolathlon.expected_native_rollouts == 404
    assert toolathlon.expected_candidates == 404
    assert toolathlon.require_completed_run is True
    assert toolathlon.trace_format == "toolathlon_legacy_unversioned"
    assert toolathlon.subagent_type_aliases == {
        "qwen_3_5_4b_non_thinking": "qwen35_4b_non_thinking"
    }
    assert spec.tokenization is not None
    assert spec.tokenization.revision == ("851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


def test_qwen35_filtered_mixed_spec_uses_source_specific_selection() -> None:
    spec = load_build_spec(
        Path(
            "data/sft/specs/"
            "decomposer_mixed_deepseek_qwen35_4b_nonthinking_"
            "v1_filtered_pass_quality_32k.yaml"
        )
    ).spec

    assert spec.spec_version == 3
    assert spec.dataset.version == (
        "v1-final-493c24c4-404-wp-r1-tool-pass-or-qgt90-or-missing-32k"
    )
    assert spec.selection.policy == "all_rewards"
    workplace, toolathlon = spec.sources
    assert workplace.selection is not None
    assert workplace.selection.policy == "exact_reward"
    assert workplace.selection.success_reward == 1.0
    assert workplace.sampling is not None
    assert workplace.sampling.max_per_task == 1
    assert workplace.sampling.seed == 42
    assert toolathlon.selection is not None
    assert toolathlon.selection.policy == "toolathlon_pass_or_quality"
    assert toolathlon.selection.minimum_check_ratio_exclusive == 0.9
    assert toolathlon.expected_native_rollouts == 404
    assert toolathlon.expected_candidates == 404
    assert toolathlon.require_completed_run is True


def test_qwen35_partial_mixed_spec_pins_snapshot_cardinality() -> None:
    spec = load_build_spec(
        Path(
            "data/sft/specs/decomposer_mixed_deepseek_qwen35_4b_"
            "nonthinking_v1_partial_3983f605_327_32k.yaml"
        )
    ).spec
    assert spec.dataset.version == "v1-partial-3983f605-327-32k"
    assert spec.selection.policy == "all_rewards"
    workplace, toolathlon = spec.sources
    assert workplace.expected_native_rollouts == 3765
    assert workplace.expected_candidates == 1255
    assert toolathlon.expected_native_rollouts == 327
    assert toolathlon.expected_candidates == 327
    assert toolathlon.require_completed_run is False
    assert spec.tokenization is not None
    assert spec.tokenization.max_tokens == 32768
