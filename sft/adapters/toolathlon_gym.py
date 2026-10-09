"""Adapter from Toolathlon-Gym traces to canonical SFT records.

It reads ``toolathlon_langgraph_v1`` collections written by ``sft/toolathlon_gym``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from ..chat_tools import build_decomposer_chat_tools
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
    validate_chat_tools,
    validate_decomposer_messages,
)
from ..snapshots import SnapshotFile
from ..toolathlon_gym.scheduler import load_launch_outcome
from .base import AdapterReadResult
from .langgraph_messages import convert_langgraph_messages

ADAPTER_VERSION = 10


def _empty_counts() -> Counter[str]:
    return Counter({reason: 0 for reason in EXCLUSION_REASONS})


def _load_json(path: Path) -> JsonObject:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected {path} to contain a JSON object.")
    return value


def snapshot_files(source_dir: Path) -> list[SnapshotFile]:
    """The files of every finished episode of a toolathlon_langgraph_v1 collection.

    An episode is finished once its evaluation exists: the collector writes
    ``result.json`` after the trace, so episodes still running are left out.
    """
    files: list[SnapshotFile] = []
    for result_path in sorted(source_dir.glob("evals/*/*/result.json")):
        episode = Path("traces") / result_path.parent.relative_to(source_dir / "evals")
        if not all(
            (source_dir / episode / name).is_file()
            for name in ("trace.json", "runtime.json")
        ):
            continue
        files.extend(
            (
                SnapshotFile((episode / "trace.json").as_posix()),
                SnapshotFile((episode / "runtime.json").as_posix()),
                SnapshotFile(result_path.relative_to(source_dir).as_posix()),
            )
        )
    return files


def read_toolathlon_gym_source(
    source: SourceSpec,
    selection: SelectionSpec,
    *,
    system_prompt: str,
    canonical_tools: Sequence[JsonObject] | None = None,
    canonical_subagent_type_ids: frozenset[str] = frozenset(),
) -> AdapterReadResult:
    """Read the finished episodes of a toolathlon_langgraph_v1 collection.

    A trace qualifies by the collector's own rule
    (``sft.toolathlon_gym.scheduler.LaunchOutcome.qualifies``). The traces store
    no tool schemas; the spec's ``native_subagent_types`` rebuild them with the
    Decomposer core.
    """
    if selection.policy != "collector_qualifies":
        raise ValueError(
            f"Source {source.id!r}: toolathlon_langgraph_v1 selects with "
            "collector_qualifies."
        )
    threshold = selection.success_threshold
    assert threshold is not None
    source_dir = source.path.resolve()
    native_type_ids = {subagent.id for subagent in source.native_subagent_types}
    tools = (
        deepcopy(list(canonical_tools))
        if canonical_tools is not None
        else validate_chat_tools(
            build_decomposer_chat_tools(
                [
                    {
                        "agent_type_id": subagent.id,
                        "description": subagent.description,
                        "assistant_id": subagent.id,
                    }
                    for subagent in source.native_subagent_types
                ]
            )
        )
    )
    result_paths = sorted(source_dir.glob("evals/*/*/result.json"))
    native_rollouts = len(result_paths)
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
        and native_rollouts != source.expected_candidates
    ):
        raise ValueError(
            f"Source {source.id!r} expected {source.expected_candidates} "
            f"candidate rollouts, found {native_rollouts}."
        )

    records: list[CanonicalRollout] = []
    counts = _empty_counts()
    run_ids: set[str] = set()
    normalized_subagent_calls = 0
    for result_path in result_paths:
        counts["rollouts"] += 1
        task = result_path.parents[1].name
        episode_id = result_path.parent.name
        episode_dir = source_dir / "traces" / task / episode_id
        try:
            result = _load_json(result_path)
            if result.get("episode_id") != episode_id or result.get("task") != task:
                raise TraceValidationError(
                    "excluded_invalid_metadata",
                    f"Result identity mismatch for {episode_id}.",
                )
            outcome = load_launch_outcome(task, str(result_path))
            if not outcome.qualifies(threshold):
                counts["excluded_reward"] += 1
                continue
            if not (episode_dir / "trace.json").is_file():
                raise TraceValidationError(
                    "excluded_missing_final_state",
                    f"Missing trace.json for {episode_id}.",
                )
            if not (episode_dir / "runtime.json").is_file():
                raise TraceValidationError(
                    "excluded_missing_materialized_input",
                    f"Missing runtime.json for {episode_id}.",
                )
            trace = _load_json(episode_dir / "trace.json")
            runtime = _load_json(episode_dir / "runtime.json")
            run_id = trace.get("run_id")
            if (
                trace.get("episode_id") != episode_id
                or trace.get("task") != task
                or trace.get("purpose") != "trace-generation"
                or not isinstance(run_id, str)
                or not run_id
            ):
                raise TraceValidationError(
                    "excluded_invalid_metadata",
                    f"Trace identity mismatch for {episode_id}.",
                )
            agents = trace.get("agents")
            if not isinstance(agents, Mapping) or not all(
                isinstance(agent, Mapping) for agent in agents.values()
            ):
                raise TraceValidationError(
                    "excluded_invalid_metadata", f"Invalid agents for {episode_id}."
                )
            undeclared = sorted(
                {str(agent.get("agent_type_id")) for agent in agents.values()}
                - native_type_ids
            )
            if undeclared:
                # The rebuilt tool schema would differ from the one the teacher saw.
                raise ValueError(
                    f"Source {source.id!r} episode {episode_id} uses undeclared "
                    "subagent types: " + ", ".join(undeclared)
                )
            task_config = runtime.get("task_config")
            if not isinstance(task_config, Mapping) or task_config.get("id") != task:
                raise TraceValidationError(
                    "excluded_invalid_metadata",
                    f"Invalid runtime task metadata for {episode_id}.",
                )

            messages = convert_langgraph_messages(trace.get("messages"), system_prompt)
            if messages[1]["content"] != task_config.get("task_str"):
                raise TraceValidationError(
                    "excluded_prompt_mismatch",
                    f"Trace/runtime prompt mismatch for {episode_id}.",
                )
            normalized_type_calls = normalize_subagent_type_ids(
                messages,
                allowed_ids=canonical_subagent_type_ids,
                aliases=source.subagent_type_aliases,
            )
            validate_decomposer_messages(
                messages,
                subagent_type_ids=canonical_subagent_type_ids,
                allow_core_errors=True,
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
                    f"Invalid repetition/attempt for {episode_id}.",
                )
            generation_config = trace.get("decomposer_generation_config")
            agent_runs = trace.get("agent_runs")
            subagent_statuses = Counter(
                str(run.get("status") or "unknown")
                for run in (
                    agent_runs.values() if isinstance(agent_runs, Mapping) else ()
                )
                if isinstance(run, Mapping)
            )
            partial = outcome.partial_score
            reward = 1.0 if outcome.strict_pass else partial.fraction
            metrics = {"reward": reward, "binary_pass": float(outcome.strict_pass)}
            if partial is not None:
                metrics["check_fraction"] = partial.fraction
            records.append(
                CanonicalRollout(
                    id=f"toolathlon_gym:{source.benchmark}:{source.id}:{episode_id}",
                    group_id=f"toolathlon_gym:{task}",
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
                        task_id=task,
                        rollout_id=f"{run_id}:r{repetition:03d}:a{attempt:03d}",
                    ),
                    outcome=CanonicalOutcome(
                        success=True, reward=reward, metrics=metrics
                    ),
                    attributes={
                        "category": "toolathlon_gym",
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "repetition": repetition,
                        "attempt": attempt,
                        # The served model: the collection moved from NVFP4 to FP8.
                        "decomposer_model": (
                            generation_config.get("model_name")
                            if isinstance(generation_config, Mapping)
                            else None
                        ),
                        "subagent_model": trace.get("agent_api_model"),
                        "needed_mcp_servers": task_config.get("needed_mcp_servers"),
                        "subagent_statuses": dict(sorted(subagent_statuses.items())),
                        **(
                            {"partial_score_source": partial.source}
                            if partial is not None
                            else {}
                        ),
                        **(
                            {
                                "subagent_type_normalization": {
                                    "tool_calls": normalized_type_calls,
                                }
                            }
                            if normalized_type_calls
                            else {}
                        ),
                    },
                )
            )
            run_ids.add(run_id)
            counts["eligible"] += 1
            normalized_subagent_calls += normalized_type_calls
        except (json.JSONDecodeError, TypeError) as error:
            trace_error = TraceValidationError("excluded_invalid_json", str(error))
            if selection.invalid_policy == "error":
                raise ValueError(f"{episode_dir}: {trace_error}") from error
            counts[trace_error.reason] += 1
        except TraceValidationError as error:
            if selection.invalid_policy == "error":
                raise ValueError(f"{episode_dir}: {error}") from error
            counts[error.reason] += 1

    strict_passes = sum(
        int(record.outcome.metrics["binary_pass"]) for record in records
    )
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
            "run_ids": sorted(run_ids),
            "native_rollouts": native_rollouts,
            "candidate_rollouts": native_rollouts,
            "sidecar_failure_records": 0,
            "tool_schema_origin": (
                "canonical_policy_interface"
                if canonical_tools is not None
                else "native_subagent_types"
            ),
            "subagent_type_normalization": {
                "aliases": dict(sorted(source.subagent_type_aliases.items())),
                "tool_calls": normalized_subagent_calls,
            },
            "selection": selection.model_dump(
                mode="json",
                exclude={"invalid_policy", "max_traces_per_prompt_per_teacher"},
            ),
            "qualifying": {
                "strict_pass": strict_passes,
                "score_above_threshold": len(records) - strict_passes,
            },
        },
        counts=counts,
    )
