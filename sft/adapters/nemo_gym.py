"""Adapter from NeMo Gym rollout sidecars to canonical Decomposer traces.

A successful rollout is kept with the manager's mistakes that the core answered
(unknown tools, malformed arguments, injected user messages), since the recovery
that follows them is the core's own behavior.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .base import (
    AdapterReadResult,
    canonical_source,
    check_native_rollouts,
    empty_counts,
)
from .langgraph_messages import convert_langgraph_messages
from ..schema import (
    CanonicalOutcome,
    CanonicalRollout,
    JsonObject,
    SelectionSpec,
    SourceSpec,
    TraceValidationError,
    canonical_json,
    normalize_response_tools,
    require_mapping,
    sha256_file,
    sha256_text,
)

ADAPTER_VERSION = 8


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


def snapshot_files(source_dir: Path) -> list[str]:
    """The result files `read_nemo_gym_source` reads."""
    files = ["rollouts.jsonl", "rollouts_materialized_inputs.jsonl"]
    if (source_dir / "rollouts_failures.jsonl").is_file():
        files.append("rollouts_failures.jsonl")
    return files


def read_nemo_gym_source(
    source: SourceSpec,
    selection: SelectionSpec,
    *,
    source_dir: Path,
    system_prompt: str,
) -> AdapterReadResult:
    """Read one immutable NeMo Gym result directory."""
    source_dir = source_dir.resolve()
    rollouts_path = source_dir / "rollouts.jsonl"
    materialized_path = source_dir / "rollouts_materialized_inputs.jsonl"
    failures_path = source_dir / "rollouts_failures.jsonl"
    if not rollouts_path.is_file():
        raise FileNotFoundError(f"Missing rollout file: {rollouts_path}")
    if not materialized_path.is_file():
        raise FileNotFoundError(f"Missing materialized-input file: {materialized_path}")

    materialized = _materialized_inputs(materialized_path)
    records: list[CanonicalRollout] = []
    counts = empty_counts()
    native_rollouts = _count_nonempty_lines(rollouts_path)
    check_native_rollouts(source, native_rollouts)

    with rollouts_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
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
                if numeric_reward != selection.success_reward:
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
                messages = convert_langgraph_messages(
                    final_state.get("messages"), system_prompt
                )
                tools = normalize_response_tools(response.get("tools"))
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
                        source=canonical_source(
                            source,
                            ADAPTER_VERSION,
                            task_id=str(task_index),
                            rollout_id=str(rollout_index),
                        ),
                        outcome=CanonicalOutcome(
                            success=True,
                            reward=numeric_reward,
                            metrics={"reward": numeric_reward},
                        ),
                        attributes={"category": category},
                    )
                )
                counts["eligible"] += 1
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
            "candidate_rollouts": native_rollouts,
            "materialized_records": len(materialized),
            "sidecar_failure_records": _count_nonempty_lines(failures_path),
            "tool_schema_origin": "response.tools",
        },
        counts=counts,
    )
