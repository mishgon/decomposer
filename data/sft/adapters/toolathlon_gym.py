"""Adapter from Toolathlon-Gym schema-v2 traces to canonical SFT records."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
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
    canonical_json,
    normalize_subagent_type_ids,
    sequentialize_parallel_spawn_calls,
    sha256_file,
    sha256_text,
    validate_chat_tools,
    validate_decomposer_messages,
)

ADAPTER_VERSION = 2
TRACE_SCHEMA_VERSION = 2
IMPORT_SCHEMA_VERSION = 1


def _empty_counts() -> Counter[str]:
    return Counter({reason: 0 for reason in EXCLUSION_REASONS})


def _load_json(path: Path) -> JsonObject:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected {path} to contain a JSON object.")
    return value


def _manifest_relative_path(value: str, manifest_path: Path) -> Path:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.name == "import_manifest.json"
        or ".." in relative.parts
        or "\\" in value
    ):
        raise ValueError(f"Unsafe imported-file path in {manifest_path}: {value!r}")
    return Path(*relative.parts)


def _validate_import(source_dir: Path) -> JsonObject:
    manifest_path = source_dir / "import_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Toolathlon import manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != IMPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Toolathlon import schema: {manifest.get('schema_version')!r}"
        )
    run_id = manifest.get("run_id")
    files = manifest.get("files")
    if not isinstance(run_id, str) or not run_id or not isinstance(files, Mapping):
        raise ValueError(f"Invalid Toolathlon import manifest: {manifest_path}")
    tracked_files: set[str] = set()
    for relative, raw_metadata in files.items():
        if not isinstance(relative, str) or not isinstance(raw_metadata, Mapping):
            raise ValueError(f"Invalid imported-file entry in {manifest_path}")
        safe_relative = _manifest_relative_path(relative, manifest_path)
        path = source_dir / safe_relative
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(source_dir)
        ):
            raise FileNotFoundError(f"Imported Toolathlon file is missing: {path}")
        if path.stat().st_size != raw_metadata.get("bytes"):
            raise ValueError(f"Imported Toolathlon file size changed: {path}")
        if sha256_file(path) != raw_metadata.get("sha256"):
            raise ValueError(f"Imported Toolathlon file checksum changed: {path}")
        tracked_files.add(safe_relative.as_posix())
    actual_files = {
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != tracked_files:
        raise ValueError(
            "Imported Toolathlon file set changed: "
            f"untracked={sorted(actual_files - tracked_files)}, "
            f"missing={sorted(tracked_files - actual_files)}"
        )
    return manifest


def _extract_teacher_reasoning(message: Mapping[str, Any]) -> str | None:
    additional = message.get("additional_kwargs")
    if not isinstance(additional, Mapping):
        return None
    reasoning = additional.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) and reasoning else None


def _convert_tool_call(raw_call: Mapping[str, Any], message_index: int) -> JsonObject:
    call_id = raw_call.get("id")
    name = raw_call.get("name")
    arguments = raw_call.get("args")
    if (
        not isinstance(call_id, str)
        or not call_id
        or name not in {"spawn_subagent", "wait"}
        or not isinstance(arguments, Mapping)
    ):
        raise TraceValidationError(
            "excluded_invalid_tool_calls",
            f"Assistant message {message_index} has an invalid tool call.",
        )
    return {
        "type": "function",
        "id": call_id,
        "function": {"name": name, "arguments": dict(arguments)},
    }


def _convert_message(wrapper: Any, index: int) -> JsonObject:
    if not isinstance(wrapper, Mapping):
        raise TraceValidationError(
            "excluded_invalid_messages", f"Trace message {index} must be an object."
        )
    message_type = wrapper.get("type")
    message = wrapper.get("data")
    if not isinstance(message, Mapping) or message.get("type") != message_type:
        raise TraceValidationError(
            "excluded_invalid_messages",
            f"Trace message {index} has inconsistent wrapper and data types.",
        )
    content = message.get("content")
    if not isinstance(content, str):
        raise TraceValidationError(
            "excluded_invalid_messages",
            f"Trace message {index} content must be a string.",
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
            f"Unsupported Toolathlon message type {message_type!r} at index {index}.",
        )
    if message.get("invalid_tool_calls"):
        raise TraceValidationError(
            "excluded_invalid_tool_calls",
            f"Assistant message {index} contains invalid_tool_calls.",
        )
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list) or not all(
        isinstance(call, Mapping) for call in raw_calls
    ):
        raise TraceValidationError(
            "excluded_invalid_tool_calls",
            f"Assistant message {index} tool_calls must be a list of objects.",
        )
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [_convert_tool_call(call, index) for call in raw_calls],
        "teacher_reasoning": _extract_teacher_reasoning(message),
    }


def _convert_messages(
    messages: Any, system_prompt: str
) -> tuple[list[JsonObject], int, int]:
    if not isinstance(messages, list) or not messages:
        raise TraceValidationError(
            "excluded_missing_final_state", "trace.messages must be a non-empty list."
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


def _failed_attempts(run_manifest: Mapping[str, Any]) -> int:
    episodes = run_manifest.get("episodes")
    if not isinstance(episodes, list):
        return 0
    return sum(
        1
        for episode in episodes
        if isinstance(episode, Mapping)
        for attempt in (episode.get("attempts") or [])
        if isinstance(attempt, Mapping) and attempt.get("status") == "failed"
    )


def _completed_trace_paths(
    source_dir: Path,
    run_id: str,
    run_manifest: Mapping[str, Any],
) -> list[Path]:
    if run_manifest.get("status") != "completed":
        raise ValueError(
            f"Toolathlon run {run_id} must be completed before SFT preparation."
        )
    episodes = run_manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Toolathlon run {run_id} has no completed episode manifest.")
    trace_paths: list[Path] = []
    seen_episode_ids: set[str] = set()
    for index, raw_episode in enumerate(episodes):
        if not isinstance(raw_episode, Mapping):
            raise ValueError(f"Toolathlon run episode {index} is not an object.")
        task = raw_episode.get("task")
        repetition = raw_episode.get("repetition")
        attempt = raw_episode.get("attempt")
        if (
            raw_episode.get("status") != "completed"
            or not isinstance(task, str)
            or not task
            or not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or repetition < 1
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 1
        ):
            raise ValueError(
                f"Toolathlon run {run_id} episode {index} is not completed with "
                "valid task/repetition/attempt metadata."
            )
        task_digest = sha256_text(task)[:8]
        episode_id = (
            f"{run_id}-{task_digest}-r{repetition:03d}-a{attempt:03d}"
        )
        if episode_id in seen_episode_ids:
            raise ValueError(f"Duplicate completed Toolathlon episode {episode_id}.")
        seen_episode_ids.add(episode_id)
        trace_path = source_dir / "traces" / task / episode_id / "trace.json"
        if not trace_path.is_file():
            raise ValueError(
                f"Completed Toolathlon episode has no imported trace: {episode_id}."
            )
        trace_paths.append(trace_path)
    return sorted(trace_paths)


def read_toolathlon_gym_source(
    source: SourceSpec,
    selection: SelectionSpec,
    *,
    system_prompt: str,
    canonical_tools: Sequence[JsonObject] | None = None,
    canonical_subagent_type_ids: frozenset[str] = frozenset(),
) -> AdapterReadResult:
    """Read one immutable, run-scoped Toolathlon-Gym import."""
    source_dir = source.path.resolve()
    import_manifest = _validate_import(source_dir)
    run_id = str(import_manifest["run_id"])
    run_manifest_path = source_dir / "runs" / run_id / "manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Missing Toolathlon run manifest: {run_manifest_path}")
    run_manifest = _load_json(run_manifest_path)
    if run_manifest.get("run_id") != run_id:
        raise ValueError(f"Toolathlon run manifest does not match {run_id}")

    all_trace_paths = sorted(source_dir.glob("traces/*/*/trace.json"))
    trace_paths = (
        _completed_trace_paths(source_dir, run_id, run_manifest)
        if source.require_completed_run
        else all_trace_paths
    )
    native_rollouts = len(trace_paths)
    if (
        source.expected_native_rollouts is not None
        and native_rollouts != source.expected_native_rollouts
    ):
        raise ValueError(
            f"Source {source.id!r} expected {source.expected_native_rollouts} "
            f"native rollouts, found {native_rollouts}."
        )
    if (
        source.expected_candidates is not None
        and len(trace_paths) != source.expected_candidates
    ):
        raise ValueError(
            f"Source {source.id!r} expected {source.expected_candidates} "
            f"candidate rollouts, found {len(trace_paths)}."
        )
    result_paths = sorted(source_dir.glob("evals/*/*/result.json"))
    result_episode_ids = {path.parent.name for path in result_paths}
    trace_episode_ids = {path.parent.name for path in trace_paths}
    records: list[CanonicalRollout] = []
    counts = _empty_counts()
    paired_records = 0
    tool_schema_hashes: set[str] = set()
    normalized_subagent_calls = 0
    legacy_schema_traces = 0

    for trace_path in trace_paths:
        counts["rollouts"] += 1
        task_from_path = trace_path.parents[1].name
        episode_from_path = trace_path.parent.name
        runtime_path = trace_path.with_name("runtime.json")
        result_path = (
            source_dir / "evals" / task_from_path / episode_from_path / "result.json"
        )
        try:
            if not result_path.is_file():
                raise TraceValidationError(
                    "excluded_missing_evaluation",
                    f"Missing result.json for {episode_from_path}.",
                )
            paired_records += 1
            result = _load_json(result_path)
            if (
                result.get("episode_id") != episode_from_path
                or result.get("task") != task_from_path
            ):
                raise TraceValidationError(
                    "excluded_invalid_metadata",
                    f"Trace/result identity mismatch for {episode_from_path}.",
                )

            reward = result.get("pass")
            if not isinstance(reward, bool):
                raise TraceValidationError(
                    "excluded_invalid_reward",
                    f"Toolathlon result.pass must be boolean for {episode_from_path}.",
                )
            numeric_reward = float(reward)
            if not math.isfinite(numeric_reward):
                raise TraceValidationError(
                    "excluded_invalid_reward",
                    f"Invalid reward for {episode_from_path}.",
                )
            if (
                selection.policy == "exact_reward"
                and numeric_reward != selection.success_reward
            ):
                counts["excluded_reward"] += 1
                continue

            if not runtime_path.is_file():
                raise TraceValidationError(
                    "excluded_missing_materialized_input",
                    f"Missing runtime.json for {episode_from_path}.",
                )
            trace = _load_json(trace_path)
            runtime = _load_json(runtime_path)
            if source.trace_format == "toolathlon_legacy_unversioned":
                if (
                    trace.get("schema_version") is not None
                    or trace.get("tools") is not None
                ):
                    raise TraceValidationError(
                        "excluded_invalid_tool_schema",
                        f"Legacy trace {episode_from_path} unexpectedly embeds a schema.",
                    )
                if canonical_tools is None:
                    raise TraceValidationError(
                        "excluded_invalid_tool_schema",
                        "Legacy Toolathlon traces require a canonical policy interface.",
                    )
                native_tools = None
                legacy_schema_traces += 1
            else:
                if trace.get("schema_version") != TRACE_SCHEMA_VERSION:
                    raise TraceValidationError(
                        "excluded_invalid_tool_schema",
                        f"Trace {episode_from_path} must use schema_version 2.",
                    )
                native_tools = validate_chat_tools(trace.get("tools"))
            if trace.get("purpose") != "trace-generation":
                raise TraceValidationError(
                    "excluded_invalid_metadata",
                    f"Trace {episode_from_path} is not a trace-generation episode.",
                )
            if (
                trace.get("run_id") != run_id
                or trace.get("episode_id") != episode_from_path
                or trace.get("task") != task_from_path
            ):
                raise TraceValidationError(
                    "excluded_invalid_metadata",
                    f"Trace/result identity mismatch for {episode_from_path}.",
                )
            task_config = runtime.get("task_config")
            if not isinstance(task_config, Mapping):
                raise TraceValidationError(
                    "excluded_missing_materialized_input",
                    f"runtime.json has no task_config for {episode_from_path}.",
                )
            task_prompt = task_config.get("task_str")
            if task_config.get("id") != task_from_path or not isinstance(
                task_prompt, str
            ):
                raise TraceValidationError(
                    "excluded_invalid_metadata",
                    f"Invalid runtime task metadata for {episode_from_path}.",
                )

            messages, normalized_messages, normalized_calls = _convert_messages(
                trace.get("messages"), system_prompt
            )
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
            if tools is None:
                raise TraceValidationError(
                    "excluded_invalid_tool_schema",
                    f"Trace {episode_from_path} has no usable tool schema.",
                )
            if messages[1]["content"] != task_prompt:
                raise TraceValidationError(
                    "excluded_prompt_mismatch",
                    f"Trace/runtime prompt mismatch for {episode_from_path}.",
                )
            repetition = trace.get("repetition")
            attempt = trace.get("attempt")
            if (
                not isinstance(repetition, int)
                or isinstance(repetition, bool)
                or repetition < 1
                or not isinstance(attempt, int)
                or isinstance(attempt, bool)
                or attempt < 1
            ):
                raise TraceValidationError(
                    "excluded_invalid_indices",
                    f"Invalid repetition/attempt for {episode_from_path}.",
                )

            subagent_statuses = (
                Counter(
                    str(run.get("status") or "unknown")
                    for run in (trace.get("subagent_runs") or {}).values()
                    if isinstance(run, Mapping)
                )
                if isinstance(trace.get("subagent_runs"), Mapping)
                else Counter()
            )
            needed_servers = task_config.get("needed_mcp_servers")
            if not isinstance(needed_servers, list) or not all(
                isinstance(item, str) for item in needed_servers
            ):
                needed_servers = []
            attributes: JsonObject = {
                "category": "toolathlon_gym",
                "run_id": run_id,
                "episode_id": episode_from_path,
                "repetition": repetition,
                "attempt": attempt,
                "decomposer_model": trace.get("decomposer_model"),
                "subagent_model": trace.get("subagent_model"),
                "needed_mcp_servers": needed_servers,
                "subagent_statuses": dict(sorted(subagent_statuses.items())),
                **(
                    {
                        "subagent_type_normalization": {
                            "tool_calls": normalized_type_calls,
                        }
                    }
                    if normalized_type_calls
                    else {}
                ),
            }
            if normalized_messages:
                attributes["parallel_spawn_normalization"] = {
                    "messages": normalized_messages,
                    "tool_calls": normalized_calls,
                }
            records.append(
                CanonicalRollout(
                    id=(
                        f"toolathlon_gym:{source.benchmark}:{source.id}:"
                        f"{episode_from_path}"
                    ),
                    group_id=f"toolathlon_gym:{task_from_path}",
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
                        task_id=task_from_path,
                        rollout_id=(f"{run_id}:r{repetition:03d}:a{attempt:03d}"),
                    ),
                    outcome=CanonicalOutcome(
                        success=numeric_reward == selection.success_reward,
                        reward=numeric_reward,
                        metrics={"reward": numeric_reward},
                    ),
                    attributes=attributes,
                )
            )
            tool_schema_hashes.add(sha256_text(canonical_json(tools)))
            counts["eligible"] += 1
            normalized_subagent_calls += normalized_type_calls
        except (json.JSONDecodeError, TypeError) as error:
            trace_error = TraceValidationError("excluded_invalid_json", str(error))
            if selection.invalid_policy == "error":
                raise ValueError(f"{trace_path}: {trace_error}") from error
            counts[trace_error.reason] += 1
        except TraceValidationError as error:
            if selection.invalid_policy == "error":
                raise ValueError(f"{trace_path}: {error}") from error
            counts[error.reason] += 1

    episodes = run_manifest.get("episodes")
    planned_episodes = len(episodes) if isinstance(episodes, list) else None
    imported_files = dict(import_manifest["files"])
    archive = import_manifest.get("archive")
    archive_sha256 = archive.get("sha256") if isinstance(archive, Mapping) else None
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
            "run_id": run_id,
            "run_status": run_manifest.get("status"),
            "planned_episodes": planned_episodes,
            "native_rollouts": native_rollouts,
            "candidate_rollouts": len(trace_paths),
            "trace_records": len(all_trace_paths),
            "ignored_attempt_trace_records": len(all_trace_paths) - len(trace_paths),
            "paired_records": paired_records,
            "unpaired_trace_records": len(trace_paths) - paired_records,
            "unpaired_evaluation_records": len(result_episode_ids - trace_episode_ids),
            "sidecar_failure_records": _failed_attempts(run_manifest),
            "archive_sha256": archive_sha256,
            "files": imported_files,
            "tool_schema_origin": (
                "canonical_policy_interface"
                if canonical_tools is not None
                else "trace.tools"
            ),
            "legacy_schema_traces": legacy_schema_traces,
            "subagent_type_normalization": {
                "aliases": dict(sorted(source.subagent_type_aliases.items())),
                "tool_calls": normalized_subagent_calls,
            },
            "eligible_tool_schema_sha256s": sorted(tool_schema_hashes),
        },
        counts=counts,
    )
