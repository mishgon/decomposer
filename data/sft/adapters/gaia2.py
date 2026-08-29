"""Strict adapter for completed GAIA2 Decomposer evaluation traces."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from .base import AdapterReadResult
from ..schema import (
    EXCLUSION_REASONS,
    CanonicalOutcome,
    CanonicalRollout,
    CanonicalSource,
    JsonObject,
    SelectionSpec,
    SourceSpec,
    TraceValidationError,
    normalize_subagent_type_ids,
    require_mapping,
    sequentialize_parallel_spawn_calls,
    sha256_file,
    validate_decomposer_messages,
)

ADAPTER_VERSION = 1


def _load_json(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_sidecar(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TraceValidationError(
            "excluded_invalid_json", f"Cannot read GAIA2 sidecar {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise TraceValidationError(
            "excluded_invalid_json", f"GAIA2 sidecar {path} must contain an object"
        )
    return value


def _load_jsonl(path: Path) -> Iterable[tuple[int, JsonObject]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}"
                )
            yield line_number, value


def _empty_counts() -> Counter[str]:
    return Counter({reason: 0 for reason in EXCLUSION_REASONS})


def _file_identity(path: Path) -> JsonObject:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _require_int(value: Any, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _load_scenario_partition(
    source: SourceSpec,
) -> tuple[set[str], set[str], JsonObject]:
    if source.gaia2 is None:
        raise AssertionError("GAIA2 source options are required")
    path = source.gaia2.split_manifest.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing GAIA2 split manifest: {path}")
    manifest = _load_json(path)
    if manifest.get("schema_version", 1) != 1:
        raise ValueError("Unsupported GAIA2 split-manifest schema")
    scenarios = manifest.get("scenarios")
    partitions = manifest.get("partitions")
    if not isinstance(scenarios, list) or not isinstance(partitions, Mapping):
        raise ValueError("GAIA2 split manifest has invalid assignments")
    train_partition = partitions.get("train")
    test_partition = partitions.get("test")
    full_partition = partitions.get("full")
    if not all(
        isinstance(partition, Mapping)
        for partition in (train_partition, test_partition, full_partition)
    ):
        raise ValueError("GAIA2 split manifest has invalid partitions")
    train_universes = train_partition.get("universes")
    test_universes = test_partition.get("universes")
    full_universes = full_partition.get("universes")
    if not all(
        isinstance(universes, list)
        for universes in (train_universes, test_universes, full_universes)
    ):
        raise ValueError("GAIA2 split manifest has invalid universe lists")
    if set(train_universes) & set(test_universes) or set(train_universes) | set(
        test_universes
    ) != set(full_universes):
        raise ValueError("GAIA2 split manifest does not isolate complete universes")
    universe_assignments = {
        **{universe: "train" for universe in train_universes},
        **{universe: "test" for universe in test_universes},
    }

    all_ids: set[str] = set()
    selected_ids: set[str] = set()
    ids_by_universe: dict[int, set[str]] = {}
    for index, raw in enumerate(scenarios):
        if not isinstance(raw, Mapping):
            raise ValueError(f"GAIA2 split scenario {index} must be an object")
        scenario_id = raw.get("scenario_id")
        universe = raw.get("universe")
        partition = raw.get("partition")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or scenario_id in all_ids
            or not isinstance(universe, int)
            or isinstance(universe, bool)
            or partition not in {"train", "test"}
        ):
            raise ValueError(f"GAIA2 split scenario {index} is invalid or duplicated")
        declared_partition = universe_assignments.get(universe)
        if declared_partition != partition:
            raise ValueError(f"GAIA2 universe {universe!r} is split across partitions")
        all_ids.add(scenario_id)
        ids_by_universe.setdefault(universe, set()).add(scenario_id)
        if partition == source.gaia2.scenario_partition:
            selected_ids.add(scenario_id)

    if len(selected_ids) != source.gaia2.expected_scenarios:
        raise ValueError(
            f"GAIA2 partition expected {source.gaia2.expected_scenarios} scenarios, "
            f"found {len(selected_ids)}"
        )
    expected_train = sum(
        len(ids_by_universe[universe])
        for universe, partition in universe_assignments.items()
        if partition == "train" and universe in ids_by_universe
    )
    expected_test = len(all_ids) - expected_train
    if (
        train_partition.get("rows") != expected_train
        or test_partition.get("rows") != expected_test
        or full_partition.get("rows") != len(all_ids)
    ):
        raise ValueError("GAIA2 split manifest summary changed")
    return (
        selected_ids,
        all_ids,
        {
            "name": manifest.get("name"),
            "schema_version": 1,
            "sha256": sha256_file(path),
            "dataset": manifest.get("dataset"),
            "partition": source.gaia2.scenario_partition,
            "scenarios": len(selected_ids),
            "all_scenarios": len(all_ids),
            "universes": sorted(
                universe
                for universe, partition in universe_assignments.items()
                if partition == source.gaia2.scenario_partition
            ),
        },
    )


def _completed_marker(source_dir: Path, name: str) -> tuple[Path, JsonObject]:
    path = source_dir / name
    if not path.is_file():
        raise ValueError(f"GAIA2 source is not terminal: missing {path}")
    marker = _load_json(path)
    if marker.get("state") != "complete":
        raise ValueError(f"GAIA2 source is not complete according to {path}")
    if marker.get("kind") not in {None, "decomposer"}:
        raise ValueError("GAIA2 completion marker is not a Decomposer run")
    if marker.get("decomposer_system_prompt_profile") not in {None, "teacher"}:
        raise ValueError("GAIA2 completion marker did not use the teacher prompt")
    return path, marker


def _binary_reward(value: Any, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) not in {0.0, 1.0}
    ):
        raise ValueError(f"{description} must be a binary numeric reward")
    return float(value)


def _native_evaluation_rows(
    source: SourceSpec,
    source_dir: Path,
    selected_ids: set[str],
    all_ids: set[str],
) -> tuple[list[JsonObject], JsonObject, dict[str, Path], JsonObject]:
    marker_path, marker = _completed_marker(source_dir, ".eval_done.json")
    output_path = source_dir / "output.jsonl"
    if not output_path.is_file():
        raise FileNotFoundError(f"Missing GAIA2 evaluation output: {output_path}")
    rows: list[JsonObject] = []
    identities: set[tuple[str, int]] = set()
    logical_numbers = set(source.gaia2.logical_rollout_numbers if source.gaia2 else ())
    for line_number, raw in _load_jsonl(output_path):
        metadata = raw.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{output_path}:{line_number} has invalid metadata")
        scenario_id = raw.get("task_id")
        if scenario_id != metadata.get("scenario_id") or scenario_id not in all_ids:
            raise ValueError(
                f"{output_path}:{line_number} has invalid scenario identity"
            )
        logical = _require_int(metadata.get("run_number"), "GAIA2 run number")
        if logical not in logical_numbers:
            raise ValueError(
                f"{output_path}:{line_number} has an unexpected run number"
            )
        identity = (str(scenario_id), logical)
        if identity in identities:
            raise ValueError(f"Duplicate GAIA2 rollout identity {identity}")
        identities.add(identity)
        reward = _binary_reward(raw.get("score"), f"{output_path}:{line_number} score")
        rows.append(
            {
                "scenario_id": scenario_id,
                "logical_rollout_number": logical,
                "native_run_number": logical,
                "reward": reward,
                "status": metadata.get("status"),
                "has_exception": metadata.get("has_exception"),
                "exception_type": metadata.get("exception_type"),
                "sidecar": f"decomposer_sidecars/{scenario_id}__run{logical}.json",
                "line_number": line_number,
            }
        )
    expected_identities = {
        (scenario_id, logical) for scenario_id in all_ids for logical in logical_numbers
    }
    if identities != expected_identities:
        raise ValueError("Completed GAIA2 evaluation is not a full scenario/run grid")
    if marker.get("num_repeats") != len(logical_numbers):
        raise ValueError("GAIA2 completion marker repeat count changed")
    metrics = marker.get("metrics")
    if isinstance(metrics, Mapping) and metrics.get("rollout_rows") != len(rows):
        raise ValueError("GAIA2 completion-marker rollout count changed")
    return (
        rows,
        marker,
        {
            ".eval_done.json": marker_path,
            "output.jsonl": output_path,
        },
        {
            "native_scenarios": len(all_ids),
            "selected_scenarios": len(selected_ids),
            "holdout_rollouts": len(rows) - len(selected_ids) * len(logical_numbers),
        },
    )


def _safe_relative_path(
    source_dir: Path, value: Any, description: str
) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} escapes the GAIA2 source")
    path = (source_dir / Path(*relative.parts)).resolve()
    try:
        path.relative_to(source_dir)
    except ValueError as error:
        raise ValueError(f"{description} escapes the GAIA2 source") from error
    return relative.as_posix(), path


def _native_trace_manifest_rows(
    source: SourceSpec,
    source_dir: Path,
    selected_ids: set[str],
) -> tuple[list[JsonObject], JsonObject, dict[str, Path], JsonObject]:
    marker_path, marker = _completed_marker(source_dir, ".trace_done.json")
    manifest_path = source_dir / "trace_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing GAIA2 trace manifest: {manifest_path}")
    rows: list[JsonObject] = []
    identities: set[tuple[str, int]] = set()
    logical_numbers = set(source.gaia2.logical_rollout_numbers if source.gaia2 else ())
    for line_number, raw in _load_jsonl(manifest_path):
        scenario_id = raw.get("scenario_id")
        logical = _require_int(
            raw.get("logical_rollout_number"), "GAIA2 logical rollout number"
        )
        native = _require_int(raw.get("native_run_number"), "GAIA2 native run number")
        if scenario_id not in selected_ids or logical not in logical_numbers:
            raise ValueError(
                f"{manifest_path}:{line_number} is outside the pinned grid"
            )
        identity = (str(scenario_id), logical)
        if identity in identities:
            raise ValueError(f"Duplicate GAIA2 rollout identity {identity}")
        identities.add(identity)
        raw_sidecar = raw.get("sidecar")
        sidecar = None
        if raw_sidecar is not None:
            sidecar, _ = _safe_relative_path(
                source_dir,
                raw_sidecar,
                f"{manifest_path}:{line_number} sidecar",
            )
        rows.append(
            {
                "scenario_id": scenario_id,
                "logical_rollout_number": logical,
                "native_run_number": native,
                "reward": _binary_reward(
                    raw.get("reward"), f"{manifest_path}:{line_number} reward"
                ),
                "status": raw.get("status"),
                "has_exception": raw.get("has_exception"),
                "exception_type": raw.get("exception_type"),
                "sidecar": sidecar,
                "line_number": line_number,
            }
        )
    expected_identities = {
        (scenario_id, logical)
        for scenario_id in selected_ids
        for logical in logical_numbers
    }
    if identities != expected_identities:
        raise ValueError(
            "Completed GAIA2 trace manifest is not the pinned task/run grid"
        )
    return (
        rows,
        marker,
        {
            ".trace_done.json": marker_path,
            "trace_manifest.jsonl": manifest_path,
        },
        {
            "native_scenarios": len(selected_ids),
            "selected_scenarios": len(selected_ids),
            "holdout_rollouts": 0,
        },
    )


def _visible_content(value: Any, description: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TraceValidationError(
            "excluded_invalid_messages", f"{description} content is invalid"
        )
    parts: list[str] = []
    for raw in value:
        if isinstance(raw, str):
            parts.append(raw)
        elif isinstance(raw, Mapping) and raw.get("type") in {
            "text",
            "output_text",
        }:
            text = raw.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif isinstance(raw, Mapping) and raw.get("type") == "refusal":
            refusal = raw.get("refusal")
            if isinstance(refusal, str):
                parts.append(refusal)
    return "".join(parts)


def _teacher_reasoning(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for raw in value:
        if not isinstance(raw, Mapping) or raw.get("type") != "reasoning":
            continue
        content = raw.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str) and part:
                    parts.append(part)
                elif isinstance(part, Mapping):
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
    return "\n".join(parts) or None


def _convert_tool_call(raw: Mapping[str, Any], index: int) -> JsonObject:
    call_id = raw.get("id")
    name = raw.get("name")
    arguments = raw.get("args")
    if (
        not isinstance(call_id, str)
        or not call_id
        or name not in {"spawn_subagent", "wait"}
        or not isinstance(arguments, Mapping)
    ):
        raise TraceValidationError(
            "excluded_invalid_tool_calls",
            f"GAIA2 assistant message {index} has an invalid tool call",
        )
    return {
        "type": "function",
        "id": call_id,
        "function": {"name": name, "arguments": dict(arguments)},
    }


def _convert_message(raw: Mapping[str, Any], index: int) -> JsonObject:
    message_type = raw.get("type")
    data = require_mapping(
        raw.get("data"), f"GAIA2 message {index} data", "excluded_invalid_messages"
    )
    content = _visible_content(data.get("content"), f"GAIA2 message {index}")
    if message_type == "human":
        return {"role": "user", "content": content}
    if message_type == "tool":
        call_id = data.get("tool_call_id")
        name = data.get("name")
        if (
            not isinstance(call_id, str)
            or not call_id
            or name not in {"spawn_subagent", "wait"}
        ):
            raise TraceValidationError(
                "excluded_invalid_tool_calls", f"GAIA2 tool message {index} is invalid"
            )
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": call_id,
            "name": name,
        }
    if message_type != "ai":
        raise TraceValidationError(
            "excluded_invalid_messages",
            f"Unsupported GAIA2 message type {message_type!r}",
        )
    if data.get("invalid_tool_calls"):
        raise TraceValidationError(
            "excluded_invalid_tool_calls", f"GAIA2 assistant message {index} is invalid"
        )
    raw_calls = data.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise TraceValidationError(
            "excluded_invalid_tool_calls",
            f"GAIA2 message {index} tool calls are invalid",
        )
    tool_calls = [
        _convert_tool_call(
            require_mapping(
                call,
                f"GAIA2 assistant message {index} tool call",
                "excluded_invalid_tool_calls",
            ),
            index,
        )
        for call in raw_calls
    ]
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "teacher_reasoning": _teacher_reasoning(data.get("content")),
    }


def _messages_from_sidecar(
    sidecar: Mapping[str, Any], system_prompt: str
) -> tuple[list[JsonObject], int, int]:
    turns = sidecar.get("turns")
    if not isinstance(turns, list) or len(turns) != 1:
        raise TraceValidationError(
            "excluded_invalid_metadata",
            "GAIA2 SFT traces must contain exactly one turn",
        )
    turn = require_mapping(turns[0], "GAIA2 turn", "excluded_invalid_metadata")
    manager = require_mapping(
        turn.get("manager"), "GAIA2 manager result", "excluded_missing_final_state"
    )
    trace = require_mapping(
        manager.get("trace"), "GAIA2 manager trace", "excluded_missing_final_state"
    )
    messages = trace.get("manager_messages")
    if not isinstance(messages, list) or not messages:
        raise TraceValidationError(
            "excluded_missing_final_state", "GAIA2 manager trace has no messages"
        )
    if not all(isinstance(message, Mapping) for message in messages):
        raise TraceValidationError(
            "excluded_invalid_messages", "GAIA2 manager messages must be objects"
        )
    converted = [
        {"role": "system", "content": system_prompt},
        *[_convert_message(message, index) for index, message in enumerate(messages)],
    ]
    normalized, normalized_messages, normalized_calls = (
        sequentialize_parallel_spawn_calls(converted)
    )
    validate_decomposer_messages(normalized)
    return normalized, normalized_messages, normalized_calls


def _validate_sidecar_identity(
    sidecar: Mapping[str, Any],
    row: Mapping[str, Any],
    marker: Mapping[str, Any],
    source: SourceSpec,
) -> JsonObject:
    scenario_id = row["scenario_id"]
    native_run = row["native_run_number"]
    if (
        sidecar.get("scenario_id") != scenario_id
        or sidecar.get("run_number") != native_run
    ):
        raise TraceValidationError(
            "excluded_invalid_indices",
            "GAIA2 sidecar identity does not match its manifest",
        )
    configuration = require_mapping(
        sidecar.get("configuration"), "GAIA2 configuration", "excluded_invalid_metadata"
    )
    if configuration.get("decomposer_system_prompt_profile") not in {None, "teacher"}:
        raise TraceValidationError(
            "excluded_invalid_metadata", "GAIA2 source did not use the teacher prompt"
        )
    models = require_mapping(
        configuration.get("model_configuration"),
        "GAIA2 model configuration",
        "excluded_invalid_metadata",
    )
    manager = require_mapping(
        models.get("manager"), "GAIA2 manager model", "excluded_invalid_metadata"
    )
    worker = require_mapping(
        models.get("subagent"), "GAIA2 worker model", "excluded_invalid_metadata"
    )
    manager_name = manager.get("served_name")
    if (
        not isinstance(manager_name, str)
        or manager_name.rsplit("/", 1)[-1] != source.teacher
        or manager.get("thinking") is not True
        or worker.get("thinking") is not False
    ):
        raise TraceValidationError(
            "excluded_invalid_metadata", "GAIA2 teacher/worker configuration changed"
        )
    decomposer_revision = sidecar.get("decomposer_revision")
    gaia2_revision = sidecar.get("gaia2_revision")
    if (
        not isinstance(decomposer_revision, str)
        or not decomposer_revision
        or not isinstance(gaia2_revision, str)
        or not gaia2_revision
        or marker.get("decomposer_commit") not in {None, decomposer_revision}
        or marker.get("gaia2_commit") not in {None, gaia2_revision}
    ):
        raise TraceValidationError(
            "excluded_invalid_metadata", "GAIA2 sidecar revisions changed"
        )
    turns = sidecar.get("turns")
    turn = turns[0] if isinstance(turns, list) and turns else None
    manager_result = turn.get("manager") if isinstance(turn, Mapping) else None
    trace = manager_result.get("trace") if isinstance(manager_result, Mapping) else None
    runtime = trace.get("runtime_context") if isinstance(trace, Mapping) else None
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("scenario_id") != scenario_id
        or runtime.get("run_number") != native_run
    ):
        raise TraceValidationError(
            "excluded_invalid_metadata", "GAIA2 runtime context identity changed"
        )
    return {
        "decomposer_revision": decomposer_revision,
        "gaia2_revision": gaia2_revision,
        "manager_served_name": manager_name,
        "worker_served_name": worker.get("served_name"),
    }


def read_gaia2_source(
    source: SourceSpec,
    selection: SelectionSpec,
    *,
    system_prompt: str,
    canonical_tools: Sequence[JsonObject] | None = None,
    canonical_subagent_type_ids: frozenset[str] = frozenset(),
) -> AdapterReadResult:
    """Read a terminal GAIA2 n-repeat evaluation or round trace release."""
    if canonical_tools is None:
        raise ValueError("GAIA2 SFT ingestion requires canonical Decomposer tools")
    if source.path is None:
        raise ValueError(f"GAIA2 source {source.id!r} has no input path")
    source_dir = source.path.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing GAIA2 source directory: {source_dir}")
    selected_ids, all_ids, split_manifest = _load_scenario_partition(source)
    if source.trace_format == "gaia2_evaluation_v1":
        native_rows, marker, source_files, layout = _native_evaluation_rows(
            source, source_dir, selected_ids, all_ids
        )
    elif source.trace_format == "gaia2_trace_manifest_v1":
        native_rows, marker, source_files, layout = _native_trace_manifest_rows(
            source, source_dir, selected_ids
        )
    else:
        raise ValueError(f"Unsupported GAIA2 trace format {source.trace_format!r}")

    native_rollouts = len(native_rows)
    candidate_rows = [row for row in native_rows if row["scenario_id"] in selected_ids]
    candidate_rollouts = len(candidate_rows)
    if source.expected_native_rollouts != native_rollouts:
        raise ValueError(
            f"Source {source.id!r} expected {source.expected_native_rollouts} native "
            f"rollouts, found {native_rollouts}"
        )
    if source.expected_candidates != candidate_rollouts:
        raise ValueError(
            f"Source {source.id!r} expected {source.expected_candidates} candidates, "
            f"found {candidate_rollouts}"
        )

    counts = _empty_counts()
    records: list[CanonicalRollout] = []
    sidecar_files: dict[str, JsonObject] = {}
    normalized_spawn_messages = 0
    normalized_spawn_calls = 0
    normalized_subagent_calls = 0
    revisions: set[tuple[Any, Any]] = set()
    reward_counts: Counter[str] = Counter()
    sidecar_failure_records = 0
    effective_selection = source.selection or selection
    for row in sorted(
        candidate_rows,
        key=lambda item: (item["scenario_id"], item["logical_rollout_number"]),
    ):
        counts["rollouts"] += 1
        reward = float(row["reward"])
        reward_counts[str(int(reward))] += 1
        if row.get("has_exception") is True:
            sidecar_failure_records += 1
        if (
            effective_selection.policy == "exact_reward"
            and reward != effective_selection.success_reward
        ):
            counts["excluded_reward"] += 1
            continue
        if effective_selection.policy not in {"exact_reward", "all_rewards"}:
            raise ValueError("GAIA2 supports only binary exact/all reward selection")
        try:
            if (
                row.get("status") != "success"
                or row.get("has_exception") is not False
                or row.get("exception_type") not in {None, ""}
            ):
                raise TraceValidationError(
                    "excluded_invalid_metadata",
                    "Selected GAIA2 rollout is not successful",
                )
            try:
                relative, sidecar_path = _safe_relative_path(
                    source_dir, row.get("sidecar"), "GAIA2 sidecar"
                )
            except ValueError as error:
                raise TraceValidationError(
                    "excluded_missing_final_state", str(error)
                ) from error
            if not sidecar_path.is_file():
                raise TraceValidationError(
                    "excluded_missing_final_state",
                    f"Missing GAIA2 sidecar {sidecar_path}",
                )
            sidecar = _load_sidecar(sidecar_path)
            identity = _validate_sidecar_identity(sidecar, row, marker, source)
            messages, normalized_messages, normalized_calls = _messages_from_sidecar(
                sidecar, system_prompt
            )
            normalized_type_calls = normalize_subagent_type_ids(
                messages,
                allowed_ids=canonical_subagent_type_ids,
                aliases=source.subagent_type_aliases,
            )
            validate_decomposer_messages(messages)
            logical = int(row["logical_rollout_number"])
            scenario_id = str(row["scenario_id"])
            records.append(
                CanonicalRollout(
                    id=(
                        f"gaia2:{source.benchmark}:{source.id}:"
                        f"{scenario_id}:r{logical:02d}"
                    ),
                    group_id=f"gaia2:{scenario_id}",
                    messages=messages,
                    tools=deepcopy(list(canonical_tools)),
                    source=CanonicalSource(
                        adapter=source.adapter,
                        adapter_version=ADAPTER_VERSION,
                        source_id=source.id,
                        benchmark=source.benchmark,
                        environment=source.environment,
                        partition=source.partition,
                        teacher=source.teacher,
                        task_id=scenario_id,
                        rollout_id=f"r{logical:02d}",
                    ),
                    outcome=CanonicalOutcome(
                        success=reward == effective_selection.success_reward,
                        reward=reward,
                        metrics={"reward": reward},
                    ),
                    attributes={
                        "category": "gaia2_execution",
                        "logical_rollout_number": logical,
                        "native_run_number": int(row["native_run_number"]),
                        "trace_format": source.trace_format,
                        "revisions": identity,
                        **(
                            {
                                "subagent_type_normalization": {
                                    "tool_calls": normalized_type_calls
                                }
                            }
                            if normalized_type_calls
                            else {}
                        ),
                        **(
                            {
                                "parallel_spawn_normalization": {
                                    "messages": normalized_messages,
                                    "tool_calls": normalized_calls,
                                }
                            }
                            if normalized_messages
                            else {}
                        ),
                    },
                )
            )
            counts["eligible"] += 1
            sidecar_files[relative] = _file_identity(sidecar_path)
            revisions.add((identity["decomposer_revision"], identity["gaia2_revision"]))
            normalized_spawn_messages += normalized_messages
            normalized_spawn_calls += normalized_calls
            normalized_subagent_calls += normalized_type_calls
        except (json.JSONDecodeError, TraceValidationError) as error:
            trace_error = (
                error
                if isinstance(error, TraceValidationError)
                else TraceValidationError("excluded_invalid_json", str(error))
            )
            if effective_selection.invalid_policy == "error":
                raise ValueError(
                    f"GAIA2 rollout {row['scenario_id']} "
                    f"r{row['logical_rollout_number']}: {trace_error}"
                ) from error
            counts[trace_error.reason] += 1

    files = {name: _file_identity(path) for name, path in source_files.items()}
    files.update(dict(sorted(sidecar_files.items())))
    return AdapterReadResult(
        records=tuple(records),
        source_manifest={
            "id": source.id,
            "adapter": source.adapter,
            "adapter_version": ADAPTER_VERSION,
            "benchmark": source.benchmark,
            "environment": source.environment,
            "partition": source.partition,
            "teacher": source.teacher,
            "trace_format": source.trace_format,
            "locator": str(source_dir),
            "files": files,
            "native_rollouts": native_rollouts,
            "candidate_rollouts": candidate_rollouts,
            "split_manifest": split_manifest,
            "layout": layout,
            "logical_rollout_numbers": list(
                source.gaia2.logical_rollout_numbers if source.gaia2 else ()
            ),
            "binary_reward_counts": dict(sorted(reward_counts.items())),
            "sidecar_failure_records": sidecar_failure_records,
            "completion": {
                key: marker.get(key)
                for key in (
                    "state",
                    "started_at",
                    "finished_at",
                    "decomposer_commit",
                    "gaia2_commit",
                )
                if key in marker
            },
            "eligible_revision_pairs": [list(pair) for pair in sorted(revisions)],
            "tool_schema_origin": "canonical_policy_interface",
            "parallel_spawn_normalization": {
                "messages": normalized_spawn_messages,
                "tool_calls": normalized_spawn_calls,
            },
            "subagent_type_normalization": {
                "aliases": dict(sorted(source.subagent_type_aliases.items())),
                "tool_calls": normalized_subagent_calls,
            },
        },
        counts=counts,
    )
