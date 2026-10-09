"""Adapter from WideSeek collections (``sft/wideseek`` on main) to SFT records.

A collection holds one ``result.json`` per attempt,
``decomposer/<task>/attempt-NNN/result.json``, naming the execution it scored.
The execution keeps the manager's LangGraph state (``trace.json``) and a log of
every model call (``model_calls/*.json``). The manager's first logged call holds
the tools and system prompt it saw.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from decomposer.prompt_profiles import resolve_decomposer_system_prompt

from ..adapters.base import AdapterReadResult
from ..adapters.langgraph_messages import convert_langgraph_messages
from ..schema import (
    EXCLUSION_REASONS,
    CanonicalOutcome,
    CanonicalRollout,
    CanonicalSource,
    JsonObject,
    SelectionSpec,
    SourceSpec,
    TraceValidationError,
    validate_chat_tools,
)
from ..snapshots import SnapshotFile

ADAPTER_VERSION = 1


def _load_json(path: Path) -> JsonObject:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected {path} to contain a JSON object.")
    return value


def _qualifies(result: Mapping[str, Any], threshold: float) -> bool:
    """The collector's success rule (``gyms/wideseek/metrics.qualifies`` on main)."""
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    score = evaluation.get("score")
    return (
        result.get("status") == "finished"
        and not result.get("cleanup_errors")
        and result.get("trace_available", True)
        and evaluation.get("status") == "scored"
        and type(score) in (int, float)
        and math.isfinite(score)
        and 0 <= score <= 1
        and (score == 1.0 or score > threshold)
    )


def _execution_dir(result_path: Path, result: Mapping[str, Any]) -> Path:
    name = result.get("execution_directory")
    if not isinstance(name, str) or not name.startswith("execution-") or "/" in name:
        raise TraceValidationError(
            "excluded_invalid_metadata",
            f"Invalid execution_directory in {result_path}.",
        )
    return result_path.parent / name


def _first_manager_call(execution_dir: Path) -> tuple[Path, JsonObject] | None:
    first: tuple[float, str, Path, JsonObject] | None = None
    for path in execution_dir.glob("model_calls/*.json"):
        call = _load_json(path)
        if call.get("role") != "decomposer":
            continue
        started_at = call.get("started_at")
        if isinstance(started_at, bool) or not isinstance(started_at, (int, float)):
            raise TraceValidationError(
                "excluded_invalid_metadata", f"Model call {path} has no started_at."
            )
        if first is None or (started_at, path.name) < first[:2]:
            first = (started_at, path.name, path, call)
    return None if first is None else (first[2], first[3])


def snapshot_files(source_dir: Path) -> list[SnapshotFile]:
    """Each finished attempt's result, trace and first manager model call.

    The collector writes ``result.json`` when an attempt ends, so attempts still
    running are left out. Executions no result names (restarted ones) are left
    out too.
    """
    files: list[SnapshotFile] = []
    for result_path in sorted(source_dir.glob("decomposer/*/attempt-*/result.json")):
        execution_dir = _execution_dir(result_path, _load_json(result_path))
        first = _first_manager_call(execution_dir)
        if first is None or not (execution_dir / "trace.json").is_file():
            continue
        files.extend(
            SnapshotFile(path.relative_to(source_dir).as_posix())
            for path in (result_path, execution_dir / "trace.json", first[0])
        )
    return files


def read_wideseek_source(
    source: SourceSpec,
    selection: SelectionSpec,
    *,
    system_prompt: str,
) -> AdapterReadResult:
    """Read the attempts the collector counts as successes.

    The manager must have seen the teacher prompt; any other logged system
    prompt stops the build, since records would pair it with the wrong prompt.
    """
    threshold = selection.success_threshold
    assert threshold is not None
    teacher_prompt = resolve_decomposer_system_prompt("teacher")
    source_dir = source.path.resolve()
    result_paths = sorted(source_dir.glob("decomposer/*/attempt-*/result.json"))
    native_rollouts = len(result_paths)
    if native_rollouts != source.expected_native_rollouts:
        raise ValueError(
            f"Source {source.id!r} expected {source.expected_native_rollouts} "
            f"native rollouts, found {native_rollouts}."
        )
    if native_rollouts != source.expected_candidates:
        raise ValueError(
            f"Source {source.id!r} expected {source.expected_candidates} "
            f"candidate rollouts, found {native_rollouts}."
        )

    records: list[CanonicalRollout] = []
    counts = Counter({reason: 0 for reason in EXCLUSION_REASONS})
    revisions: set[str] = set()
    for result_path in result_paths:
        counts["rollouts"] += 1
        task_id = result_path.parents[1].name
        try:
            result = _load_json(result_path)
            attempt = result.get("attempt")
            if (
                result.get("task_id") != task_id
                or result.get("mode") != "decomposer"
                or isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or result_path.parent.name != f"attempt-{attempt:03d}"
            ):
                raise TraceValidationError(
                    "excluded_invalid_metadata",
                    f"Result identity mismatch in {result_path}.",
                )
            if not _qualifies(result, threshold):
                counts["excluded_reward"] += 1
                continue
            execution_dir = _execution_dir(result_path, result)
            if not (execution_dir / "trace.json").is_file():
                raise TraceValidationError(
                    "excluded_missing_final_state",
                    f"Missing trace.json in {execution_dir}.",
                )
            first = _first_manager_call(execution_dir)
            if first is None:
                raise TraceValidationError(
                    "excluded_invalid_tool_schema",
                    f"No manager model call in {execution_dir}.",
                )
            call = first[1]
            logged_system = call.get("system_message")
            if (
                not isinstance(logged_system, Mapping)
                or logged_system.get("content") != teacher_prompt
            ):
                raise ValueError(
                    f"Source {source.id!r} {execution_dir.name}: the manager's "
                    "logged system prompt is not the teacher prompt."
                )
            native_tools = validate_chat_tools(call.get("tools"))
            trace = _load_json(execution_dir / "trace.json")
            messages = convert_langgraph_messages(trace.get("messages"), system_prompt)
            logged_tasks = [
                message.get("content")
                for message in call.get("messages") or []
                if isinstance(message, Mapping) and message.get("type") == "human"
            ]
            if logged_tasks[:1] != [messages[1]["content"]]:
                raise TraceValidationError(
                    "excluded_prompt_mismatch",
                    f"Trace/model-call task mismatch in {execution_dir}.",
                )
            score = float(result["evaluation"]["score"])
            generation = call.get("generation")
            records.append(
                CanonicalRollout(
                    id=(
                        f"wideseek:{source.benchmark}:{source.id}:{task_id}:"
                        f"attempt-{attempt:03d}"
                    ),
                    group_id=f"wideseek:{task_id}",
                    messages=messages,
                    tools=native_tools,
                    source=CanonicalSource(
                        adapter=source.adapter,
                        adapter_version=ADAPTER_VERSION,
                        source_id=source.id,
                        benchmark=source.benchmark,
                        environment=source.environment,
                        partition=source.partition,
                        teacher=source.teacher,
                        task_id=task_id,
                        rollout_id=f"attempt-{attempt:03d}/{execution_dir.name}",
                    ),
                    outcome=CanonicalOutcome(
                        success=True, reward=score, metrics={"reward": score}
                    ),
                    attributes={
                        "category": "wideseek",
                        "attempt": attempt,
                        "execution": execution_dir.name,
                        # The served model: the collection moved from NVFP4 to FP8.
                        "decomposer_model": (
                            generation.get("model_name")
                            if isinstance(generation, Mapping)
                            else None
                        ),
                        "evaluation_metric": result["evaluation"].get("metric"),
                    },
                )
            )
            if isinstance(result.get("revision"), str):
                revisions.add(result["revision"])
            counts["eligible"] += 1
        except (json.JSONDecodeError, TypeError) as error:
            trace_error = TraceValidationError("excluded_invalid_json", str(error))
            if selection.invalid_policy == "error":
                raise ValueError(f"{result_path}: {trace_error}") from error
            counts[trace_error.reason] += 1
        except TraceValidationError as error:
            if selection.invalid_policy == "error":
                raise ValueError(f"{result_path}: {error}") from error
            counts[error.reason] += 1

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
            "collector_revisions": sorted(revisions),
            "native_rollouts": native_rollouts,
            "candidate_rollouts": native_rollouts,
            "sidecar_failure_records": 0,
            "tool_schema_origin": "first_manager_model_call",
            "selection": selection.model_dump(
                mode="json",
                exclude={"invalid_policy", "max_traces_per_prompt_per_teacher"},
            ),
        },
        counts=counts,
    )
