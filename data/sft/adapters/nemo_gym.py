"""Adapter from NeMo Gym rollout sidecars to canonical Decomposer traces."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
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
    canonical_json,
    normalize_subagent_type_ids,
    normalize_response_tools,
    require_mapping,
    sequentialize_parallel_spawn_calls,
    sha256_file,
    sha256_text,
    validate_decomposer_messages,
)

ADAPTER_VERSION = 3


def _canonical_prompt_input(value: Any) -> str:
    """Canonicalize equivalent Responses API and materialized chat messages."""
    if not isinstance(value, list):
        return canonical_json(value)
    normalized = []
    for item in value:
        if not isinstance(item, Mapping):
            normalized.append(item)
        elif (
            item.get("type") in (None, "message")
            and "role" in item
            and "content" in item
        ):
            normalized.append({"role": item["role"], "content": item["content"]})
        else:
            normalized.append(dict(item))
    return canonical_json(normalized)


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
                    f"Expected a JSON object in {path} at line {line_number}."
                )
            yield line_number, value


def _extract_teacher_reasoning(message: Mapping[str, Any]) -> str | None:
    response_metadata = message.get("response_metadata")
    if not isinstance(response_metadata, Mapping):
        return None
    response = response_metadata.get("nemo_gym_response")
    if not isinstance(response, Mapping):
        return None
    output = response.get("output")
    if not isinstance(output, list):
        return None
    parts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "reasoning":
            continue
        content = item.get("content")
        if isinstance(content, str):
            if content:
                parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, Mapping)
                    and isinstance(part.get("text"), str)
                    and part["text"]
                ):
                    parts.append(part["text"])
    return "\n".join(parts) or None


def _convert_tool_call(
    raw_tool_call: Mapping[str, Any], message_index: int
) -> JsonObject:
    call_id = raw_tool_call.get("id")
    name = raw_tool_call.get("name")
    arguments = raw_tool_call.get("args")
    if (
        not isinstance(call_id, str)
        or not call_id
        or name not in {"spawn_subagent", "wait"}
        or not isinstance(arguments, Mapping)
    ):
        raise TraceValidationError(
            "excluded_invalid_tool_calls",
            f"Assistant message {message_index} has an invalid name or arguments.",
        )
    return {
        "type": "function",
        "id": call_id,
        "function": {"name": name, "arguments": dict(arguments)},
    }


def _convert_message(message: Mapping[str, Any], index: int) -> JsonObject:
    message_type = message.get("type")
    content = message.get("content")
    if not isinstance(content, str):
        raise TraceValidationError(
            "excluded_invalid_messages",
            f"final_state.messages[{index}].content must be a string.",
        )
    if message_type == "human":
        return {"role": "user", "content": content}
    if message_type == "tool":
        call_id = message.get("tool_call_id")
        name = message.get("name")
        if (
            not isinstance(call_id, str)
            or not call_id
            or name not in {"spawn_subagent", "wait"}
        ):
            raise TraceValidationError(
                "excluded_invalid_tool_calls",
                f"Tool message {index} has invalid identity.",
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
            f"Unsupported final-state message type {message_type!r} at index {index}.",
        )
    if message.get("invalid_tool_calls"):
        raise TraceValidationError(
            "excluded_invalid_tool_calls",
            f"Assistant message {index} contains invalid_tool_calls.",
        )
    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        raise TraceValidationError(
            "excluded_invalid_tool_calls",
            f"Assistant message {index} tool_calls must be a list.",
        )
    tool_calls = [
        _convert_tool_call(
            require_mapping(
                raw_tool_call,
                f"assistant message {index} tool call",
                "excluded_invalid_tool_calls",
            ),
            index,
        )
        for raw_tool_call in raw_tool_calls
    ]
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "teacher_reasoning": _extract_teacher_reasoning(message),
    }


def _convert_messages(
    messages: Any, system_prompt: str
) -> tuple[list[JsonObject], int, int]:
    if not isinstance(messages, list) or not messages:
        raise TraceValidationError(
            "excluded_missing_final_state",
            "final_state.messages must be a non-empty list.",
        )
    if not all(isinstance(message, Mapping) for message in messages):
        raise TraceValidationError(
            "excluded_invalid_messages", "Every final-state message must be an object."
        )
    if messages[0].get("type") != "human":
        raise TraceValidationError(
            "excluded_invalid_messages", "A rollout must start with a human message."
        )
    if messages[-1].get("type") != "ai":
        raise TraceValidationError(
            "excluded_empty_training_target",
            "A rollout must end with an assistant message.",
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


def _materialized_inputs(path: Path) -> dict[tuple[int, int], JsonObject]:
    inputs: dict[tuple[int, int], JsonObject] = {}
    for line_number, record in _load_jsonl(path):
        task_index = record.get("_ng_task_index")
        rollout_index = record.get("_ng_rollout_index")
        if (
            not isinstance(task_index, int)
            or isinstance(task_index, bool)
            or not isinstance(rollout_index, int)
            or isinstance(rollout_index, bool)
        ):
            raise ValueError(
                f"Invalid task/rollout index in {path} at line {line_number}."
            )
        key = (task_index, rollout_index)
        if key in inputs:
            raise ValueError(f"Duplicate materialized input key {key} in {path}.")
        inputs[key] = record
    return inputs


def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as file:
        return sum(bool(line.strip()) for line in file)


def _sample_rollout_line_numbers(
    path: Path,
    source: SourceSpec,
) -> tuple[set[int], JsonObject]:
    sampling = source.sampling
    if sampling is None:
        raise AssertionError("Sampling configuration is required.")
    groups: dict[int, list[tuple[int, int, str]]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rollout = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Cannot sample malformed JSON in {path} at line {line_number}: "
                    f"{error}"
                ) from error
            if not isinstance(rollout, Mapping):
                raise ValueError(
                    f"Cannot sample a non-object rollout in {path} at line "
                    f"{line_number}."
                )
            task_index = rollout.get("_ng_task_index")
            rollout_index = rollout.get("_ng_rollout_index")
            if (
                not isinstance(task_index, int)
                or isinstance(task_index, bool)
                or not isinstance(rollout_index, int)
                or isinstance(rollout_index, bool)
            ):
                raise ValueError(
                    f"Cannot sample invalid task/rollout indices in {path} at "
                    f"line {line_number}."
                )
            native_id = (
                f"nemo_gym:{source.benchmark}:{source.id}:"
                f"{task_index}:{rollout_index}"
            )
            groups.setdefault(task_index, []).append(
                (rollout_index, line_number, native_id)
            )

    if len(groups) != sampling.expected_tasks:
        raise ValueError(
            f"Source {source.id!r} expected {sampling.expected_tasks} tasks before "
            f"sampling, found {len(groups)}."
        )
    selected: set[int] = set()
    for task_index, candidates in groups.items():
        rollout_indices = [rollout_index for rollout_index, _, _ in candidates]
        if (
            len(candidates) != sampling.expected_rollouts_per_task
            or len(rollout_indices) != len(set(rollout_indices))
        ):
            raise ValueError(
                f"Source {source.id!r} task {task_index} expected "
                f"{sampling.expected_rollouts_per_task} unique rollouts, found "
                f"{len(candidates)}."
            )
        ranked = sorted(
            candidates,
            key=lambda item: sha256_text(f"{sampling.seed}\0{item[2]}"),
        )
        selected.update(
            line_number
            for _rollout_index, line_number, _native_id in ranked[
                : sampling.max_per_task
            ]
        )
    native_rollouts = sum(len(candidates) for candidates in groups.values())
    return selected, {
        "strategy": sampling.strategy,
        "seed": sampling.seed,
        "max_per_task": sampling.max_per_task,
        "tasks": len(groups),
        "expected_rollouts_per_task": sampling.expected_rollouts_per_task,
        "native_rollouts": native_rollouts,
        "candidates": len(selected),
        "not_selected": native_rollouts - len(selected),
    }


def _empty_counts() -> Counter[str]:
    return Counter({reason: 0 for reason in EXCLUSION_REASONS})


def read_nemo_gym_source(
    source: SourceSpec,
    selection: SelectionSpec,
    *,
    system_prompt: str,
    canonical_tools: Sequence[JsonObject] | None = None,
    canonical_subagent_type_ids: frozenset[str] = frozenset(),
) -> AdapterReadResult:
    """Read one immutable NeMo Gym result directory."""
    source_dir = source.path.resolve()
    rollouts_path = source_dir / "rollouts.jsonl"
    materialized_path = source_dir / "rollouts_materialized_inputs.jsonl"
    failures_path = source_dir / "rollouts_failures.jsonl"
    if not rollouts_path.is_file():
        raise FileNotFoundError(f"Missing rollout file: {rollouts_path}")
    if not materialized_path.is_file():
        raise FileNotFoundError(f"Missing materialized-input file: {materialized_path}")

    materialized = _materialized_inputs(materialized_path)
    records: list[CanonicalRollout] = []
    counts = _empty_counts()
    native_rollouts = _count_nonempty_lines(rollouts_path)
    if (
        source.expected_native_rollouts is not None
        and native_rollouts != source.expected_native_rollouts
    ):
        raise ValueError(
            f"Source {source.id!r} expected {source.expected_native_rollouts} "
            f"native rollouts, found {native_rollouts}."
        )
    selected_line_numbers: set[int] | None = None
    sampling_manifest: JsonObject | None = None
    if source.sampling is not None:
        selected_line_numbers, sampling_manifest = _sample_rollout_line_numbers(
            rollouts_path, source
        )
    candidate_rollouts = (
        native_rollouts
        if selected_line_numbers is None
        else len(selected_line_numbers)
    )
    if (
        source.expected_candidates is not None
        and candidate_rollouts != source.expected_candidates
    ):
        raise ValueError(
            f"Source {source.id!r} expected {source.expected_candidates} "
            f"candidate rollouts, found {candidate_rollouts}."
        )
    normalized_subagent_calls = 0

    with rollouts_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            if (
                selected_line_numbers is not None
                and line_number not in selected_line_numbers
            ):
                continue
            counts["rollouts"] += 1
            try:
                rollout = json.loads(line)
                if not isinstance(rollout, Mapping):
                    raise TraceValidationError(
                        "excluded_invalid_json",
                        "A rollout line must contain a JSON object.",
                    )
                reward = rollout.get("reward")
                try:
                    numeric_reward = float(reward)
                except (TypeError, ValueError, OverflowError):
                    numeric_reward = math.nan
                if (
                    not isinstance(reward, (int, float))
                    or isinstance(reward, bool)
                    or not math.isfinite(numeric_reward)
                ):
                    raise TraceValidationError(
                        "excluded_invalid_reward", "A rollout reward must be numeric."
                    )
                if (
                    selection.policy == "exact_reward"
                    and numeric_reward != selection.success_reward
                ):
                    counts["excluded_reward"] += 1
                    continue

                task_index = rollout.get("_ng_task_index")
                rollout_index = rollout.get("_ng_rollout_index")
                if (
                    not isinstance(task_index, int)
                    or isinstance(task_index, bool)
                    or not isinstance(rollout_index, int)
                    or isinstance(rollout_index, bool)
                ):
                    raise TraceValidationError(
                        "excluded_invalid_indices", "Invalid task/rollout index."
                    )
                input_key = (task_index, rollout_index)
                if input_key not in materialized:
                    raise TraceValidationError(
                        "excluded_missing_materialized_input",
                        f"No materialized input for task/rollout {input_key}.",
                    )
                materialized_input = materialized[input_key]
                agent_ref = rollout.get("agent_ref")
                if (
                    not isinstance(agent_ref, Mapping)
                    or agent_ref.get("name") != "decomposer"
                ):
                    raise TraceValidationError(
                        "excluded_invalid_agent_ref",
                        "agent_ref.name must be decomposer.",
                    )
                rollout_params = require_mapping(
                    rollout.get("responses_create_params"),
                    "responses_create_params",
                    "excluded_prompt_mismatch",
                )
                materialized_params = require_mapping(
                    materialized_input.get("responses_create_params"),
                    "materialized responses_create_params",
                    "excluded_prompt_mismatch",
                )
                original_input = rollout_params.get("input")
                if _canonical_prompt_input(original_input) != _canonical_prompt_input(
                    materialized_params.get("input")
                ):
                    raise TraceValidationError(
                        "excluded_prompt_mismatch",
                        f"Rollout/materialized prompt mismatch for {input_key}.",
                    )
                group_id = sha256_text(_canonical_prompt_input(original_input))
                final_state = require_mapping(
                    rollout.get("final_state"),
                    "final_state",
                    "excluded_missing_final_state",
                )
                response = require_mapping(
                    rollout.get("response"), "response", "excluded_invalid_tool_schema"
                )
                messages, normalized_messages, normalized_calls = _convert_messages(
                    final_state.get("messages"), system_prompt
                )
                native_tools = normalize_response_tools(response.get("tools"))
                normalized_type_calls = normalize_subagent_type_ids(
                    messages,
                    allowed_ids=canonical_subagent_type_ids,
                    aliases=source.subagent_type_aliases,
                )
                validate_decomposer_messages(messages)
                tools = (
                    deepcopy(list(canonical_tools))
                    if canonical_tools is not None
                    else native_tools
                )
                category = materialized_input.get("category")
                environment = materialized_input.get("environment_name")
                if not isinstance(category, str) or not category:
                    raise TraceValidationError(
                        "excluded_invalid_metadata",
                        f"Missing category for {input_key}.",
                    )
                if environment != source.environment:
                    raise TraceValidationError(
                        "excluded_invalid_metadata",
                        f"Expected environment {source.environment!r}, got {environment!r}.",
                    )
                records.append(
                    CanonicalRollout(
                        id=(
                            f"nemo_gym:{source.benchmark}:{source.id}:"
                            f"{task_index}:{rollout_index}"
                        ),
                        group_id=group_id,
                        messages=messages,
                        tools=tools,
                        source=CanonicalSource(
                            adapter=source.adapter,
                            adapter_version=ADAPTER_VERSION,
                            source_id=source.id,
                            benchmark=source.benchmark,
                            environment=source.environment,
                            partition=source.partition,
                            teacher=source.teacher,
                            task_id=str(task_index),
                            rollout_id=str(rollout_index),
                        ),
                        outcome=CanonicalOutcome(
                            success=numeric_reward == selection.success_reward,
                            reward=numeric_reward,
                            metrics={"reward": numeric_reward},
                        ),
                        attributes={
                            "category": category,
                            **(
                                {
                                    "subagent_type_normalization": {
                                        "tool_calls": normalized_type_calls,
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
                normalized_subagent_calls += normalized_type_calls
            except json.JSONDecodeError as error:
                trace_error = TraceValidationError("excluded_invalid_json", str(error))
                if selection.invalid_policy == "error":
                    raise ValueError(
                        f"{rollouts_path}:{line_number}: {trace_error}"
                    ) from error
                counts[trace_error.reason] += 1
            except TraceValidationError as error:
                if selection.invalid_policy == "error":
                    raise ValueError(
                        f"{rollouts_path}:{line_number}: {error}"
                    ) from error
                counts[error.reason] += 1

    files: JsonObject = {
        "rollouts.jsonl": {
            "bytes": rollouts_path.stat().st_size,
            "sha256": sha256_file(rollouts_path),
        },
        "rollouts_materialized_inputs.jsonl": {
            "bytes": materialized_path.stat().st_size,
            "sha256": sha256_file(materialized_path),
        },
    }
    if failures_path.exists():
        files["rollouts_failures.jsonl"] = {
            "bytes": failures_path.stat().st_size,
            "sha256": sha256_file(failures_path),
        }
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
            "locator": str(source_dir),
            "files": files,
            "native_rollouts": native_rollouts,
            "candidate_rollouts": candidate_rollouts,
            "sampling": sampling_manifest,
            "materialized_records": len(materialized),
            "sidecar_failure_records": _count_nonempty_lines(failures_path),
            "tool_schema_origin": (
                "canonical_policy_interface"
                if canonical_tools is not None
                else "response.tools"
            ),
            "subagent_type_normalization": {
                "aliases": dict(sorted(source.subagent_type_aliases.items())),
                "tool_calls": normalized_subagent_calls,
            },
        },
        counts=counts,
    )
