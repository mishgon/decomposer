"""Canonical records and strict build specifications for Decomposer SFT."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from decomposer.prompts import EARLY_RESPONSE_ERROR, EMPTY_RESPONSE_ERROR

JsonObject = dict[str, Any]
InvalidPolicy = Literal["exclude", "error"]
SourcePartition = Literal["train", "validation", "test"]
SelectionPolicy = Literal[
    "exact_reward",
    "all_rewards",
    "toolathlon_pass_or_quality",
    "toolathlon_pass_or_quality_inclusive",
    "collector_qualifies",
]

CANONICAL_SCHEMA_VERSION = 1
MANIFEST_FORMAT_VERSION = 3

EXCLUSION_REASONS = (
    "excluded_reward",
    "excluded_quality",
    "excluded_invalid_json",
    "excluded_invalid_reward",
    "excluded_invalid_indices",
    "excluded_invalid_agent_ref",
    "excluded_missing_materialized_input",
    "excluded_missing_evaluation",
    "excluded_prompt_mismatch",
    "excluded_missing_final_state",
    "excluded_empty_training_target",
    "excluded_invalid_tool_schema",
    "excluded_invalid_tool_calls",
    "excluded_invalid_messages",
    "excluded_invalid_metadata",
    "excluded_prompt_teacher_cap",
    "excluded_legacy_tool_interface",
)

# The Decomposer manager's tool interface. Canonical SFT records use exactly these
# tools, with assistant messages kept as emitted, several calls included.
DECOMPOSER_TOOL_NAMES = frozenset({"new", "fork", "run", "wait"})
# String parameters each tool requires; there are no optional parameters.
DECOMPOSER_TOOL_PARAMETERS: dict[str, tuple[str, ...]] = {
    "new": ("agent_type_id",),
    "fork": ("agent_id",),
    "run": ("agent_id", "prompt"),
    "wait": (),
}
# Tools of the retired spawn_subagent/wait core. Preparation accepts only the
# new/fork/run/wait interface and excludes traces that use these tools.
LEGACY_DECOMPOSER_TOOL_NAMES = frozenset({"spawn_subagent"})
# Exact user messages the core injects when the manager answers before collecting
# every run, or answers with empty text; the manager then continues.
CORE_USER_MESSAGES = frozenset({EARLY_RESPONSE_ERROR, EMPTY_RESPONSE_ERROR})

# Collection formats whose sources select with collector_qualifies.
COLLECTION_TRACE_FORMATS = frozenset(
    {"toolathlon_langgraph_v1", "wideseek_langgraph_v1"}
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SNAPSHOT_REFERENCE = re.compile(r"^sha256:[0-9a-f]{64}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetIdentity(StrictModel):
    id: str
    version: str

    @field_validator("id", "version")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(
                "must start with a lowercase letter or digit and contain only "
                "lowercase letters, digits, dots, underscores, or hyphens"
            )
        return value


class SubagentInterfaceSpec(StrictModel):
    id: str
    description: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("subagent type ID must be a lowercase identifier")
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("subagent type description must be non-empty")
        return value


class PolicySpec(StrictModel):
    id: str
    # ``system_prompt`` is retained solely so immutable legacy build specs keep
    # loading. New specs select an explicit shared prompt profile.
    system_prompt: Literal["decomposer_default"] | None = None
    system_prompt_profile: Literal["student", "teacher"] | None = None
    subagent_types: tuple[SubagentInterfaceSpec, ...] = ()

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("policy.id must be a lowercase identifier")
        return value

    @model_validator(mode="after")
    def validate_subagent_types(self) -> "PolicySpec":
        if self.system_prompt is not None and self.system_prompt_profile is not None:
            raise ValueError(
                "policy.system_prompt and policy.system_prompt_profile are mutually "
                "exclusive"
            )
        ids = [subagent.id for subagent in self.subagent_types]
        if len(ids) != len(set(ids)):
            raise ValueError("policy.subagent_types IDs must be unique")
        return self

    @property
    def resolved_system_prompt_profile(self) -> Literal["student", "teacher"]:
        # Legacy ``decomposer_default`` keeps its historical student prompt; a
        # spec that names no prompt trains with the teacher's own prompt.
        if self.system_prompt_profile is not None:
            return self.system_prompt_profile
        if self.system_prompt == "decomposer_default":
            return "student"
        return "teacher"


class SourceSamplingSpec(StrictModel):
    strategy: Literal["task_hash"] = "task_hash"
    seed: int = 42
    max_per_task: int = 1
    expected_tasks: int
    expected_rollouts_per_task: int

    @field_validator("max_per_task", "expected_tasks", "expected_rollouts_per_task")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if isinstance(value, bool) or value <= 0:
            raise ValueError("source sampling counts must be positive integers")
        return value

    @model_validator(mode="after")
    def validate_limit(self) -> "SourceSamplingSpec":
        if self.max_per_task > self.expected_rollouts_per_task:
            raise ValueError(
                "sampling.max_per_task must not exceed expected_rollouts_per_task"
            )
        return self


class SourceSelectionSpec(StrictModel):
    policy: SelectionPolicy = "exact_reward"
    success_reward: float = 1.0
    minimum_check_ratio_exclusive: float | None = None
    minimum_check_ratio_inclusive: float | None = None
    # collector_qualifies keeps what the trace collector itself counted as a
    # success: a strict pass, or a score above this threshold.
    success_threshold: float | None = None

    @field_validator("success_reward")
    @classmethod
    def validate_reward(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("selection.success_reward must be finite")
        return value

    @model_validator(mode="after")
    def validate_quality_threshold(self) -> "SourceSelectionSpec":
        if self.policy == "toolathlon_pass_or_quality":
            threshold = self.minimum_check_ratio_exclusive
            if threshold is None or not math.isfinite(threshold):
                raise ValueError(
                    "toolathlon_pass_or_quality requires a finite "
                    "minimum_check_ratio_exclusive"
                )
            if not 0.0 <= threshold < 1.0:
                raise ValueError(
                    "minimum_check_ratio_exclusive must be at least 0 and less than 1"
                )
        elif self.minimum_check_ratio_exclusive is not None:
            raise ValueError(
                "minimum_check_ratio_exclusive is only valid for "
                "toolathlon_pass_or_quality"
            )
        if self.policy == "toolathlon_pass_or_quality_inclusive":
            threshold = self.minimum_check_ratio_inclusive
            if threshold is None or not math.isfinite(threshold):
                raise ValueError(
                    "toolathlon_pass_or_quality_inclusive requires a finite "
                    "minimum_check_ratio_inclusive"
                )
            if not 0.0 < threshold <= 1.0:
                raise ValueError(
                    "minimum_check_ratio_inclusive must be greater than 0 "
                    "and at most 1"
                )
        elif self.minimum_check_ratio_inclusive is not None:
            raise ValueError(
                "minimum_check_ratio_inclusive is only valid for "
                "toolathlon_pass_or_quality_inclusive"
            )
        if self.policy == "collector_qualifies":
            threshold = self.success_threshold
            if threshold is None or not math.isfinite(threshold):
                raise ValueError(
                    "collector_qualifies requires a finite success_threshold"
                )
            if not 0.0 <= threshold < 1.0:
                raise ValueError(
                    "success_threshold must be at least 0 and less than 1"
                )
        elif self.success_threshold is not None:
            raise ValueError("success_threshold is only valid for collector_qualifies")
        return self


class SelectionSpec(SourceSelectionSpec):
    invalid_policy: InvalidPolicy = "exclude"
    max_traces_per_prompt_per_teacher: int | None = None

    @field_validator("max_traces_per_prompt_per_teacher")
    @classmethod
    def validate_cap(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError(
                "selection.max_traces_per_prompt_per_teacher must be positive"
            )
        return value


class SourceSpec(StrictModel):
    id: str
    adapter: Literal["nemo_gym", "toolathlon_gym", "wideseek"]
    path: Path | None = None
    # spec_version 4 names each source by its snapshot digest (``sha256:<hex>``).
    snapshot: str | None = None
    benchmark: str
    environment: str
    partition: SourcePartition
    teacher: str
    trace_format: Literal[
        "native",
        "toolathlon_legacy_unversioned",
        "toolathlon_langgraph_v1",
        "wideseek_langgraph_v1",
    ] = "native"
    # The subagent types a toolathlon_langgraph_v1 collection offered the
    # manager. Its traces store no tool schemas, so the builder rebuilds them
    # from these types with the Decomposer core.
    native_subagent_types: tuple[SubagentInterfaceSpec, ...] = ()
    subagent_type_aliases: dict[str, str] = Field(default_factory=dict)
    sampling: SourceSamplingSpec | None = None
    selection: SourceSelectionSpec | None = None
    expected_native_rollouts: int | None = None
    expected_candidates: int | None = None
    require_completed_run: bool = False

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("source.id must be a lowercase identifier")
        return value

    @field_validator("benchmark", "environment", "teacher")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source string fields must be non-empty")
        return value

    @field_validator("snapshot")
    @classmethod
    def validate_snapshot(cls, value: str | None) -> str | None:
        if value is not None and not _SNAPSHOT_REFERENCE.fullmatch(value):
            raise ValueError("source.snapshot must look like sha256:<64 hex>")
        return value

    @field_validator("expected_native_rollouts", "expected_candidates")
    @classmethod
    def validate_expected_count(cls, value: int | None) -> int | None:
        if value is not None and (isinstance(value, bool) or value <= 0):
            raise ValueError("expected source counts must be positive integers")
        return value

    @field_validator("subagent_type_aliases")
    @classmethod
    def validate_aliases(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not isinstance(source_id, str)
            or not source_id.strip()
            or not isinstance(target_id, str)
            or not target_id.strip()
            for source_id, target_id in value.items()
        ):
            raise ValueError("subagent type aliases must map non-empty strings")
        return value

    @model_validator(mode="after")
    def validate_adapter_options(self) -> "SourceSpec":
        if (
            self.trace_format == "toolathlon_legacy_unversioned"
            and self.adapter != "toolathlon_gym"
        ):
            raise ValueError(
                "toolathlon_legacy_unversioned is only valid for Toolathlon sources"
            )
        if (
            self.trace_format == "toolathlon_langgraph_v1"
            and self.adapter != "toolathlon_gym"
        ):
            raise ValueError(
                "toolathlon_langgraph_v1 is only valid for Toolathlon sources"
            )
        if (self.trace_format == "wideseek_langgraph_v1") != (
            self.adapter == "wideseek"
        ):
            raise ValueError(
                "WideSeek sources, and only they, use wideseek_langgraph_v1"
            )
        if bool(self.native_subagent_types) != (
            self.trace_format == "toolathlon_langgraph_v1"
        ):
            raise ValueError(
                "toolathlon_langgraph_v1 sources, and only they, declare "
                "native_subagent_types"
            )
        if self.sampling is not None and self.adapter != "nemo_gym":
            raise ValueError("source task sampling is only supported for NeMo Gym")
        if self.require_completed_run and self.adapter != "toolathlon_gym":
            raise ValueError("require_completed_run is only valid for Toolathlon")
        if (
            self.selection is not None
            and self.selection.policy
            in {
                "toolathlon_pass_or_quality",
                "toolathlon_pass_or_quality_inclusive",
            }
            and self.adapter != "toolathlon_gym"
        ):
            raise ValueError(
                "toolathlon_pass_or_quality is only valid for Toolathlon sources"
            )
        if self.sampling is not None and self.expected_native_rollouts is not None:
            expected = (
                self.sampling.expected_tasks * self.sampling.expected_rollouts_per_task
            )
            if self.expected_native_rollouts != expected:
                raise ValueError(
                    "expected_native_rollouts must match the sampling task layout"
                )
        if self.sampling is not None and self.expected_candidates is not None:
            expected = self.sampling.expected_tasks * self.sampling.max_per_task
            if self.expected_candidates != expected:
                raise ValueError(
                    "expected_candidates must match the sampling task layout"
                )
        return self


class SplitSpec(StrictModel):
    strategy: Literal["prompt_fixed", "preserve"]
    seed: int = 42
    validation_fraction: float | None = None

    @model_validator(mode="after")
    def validate_strategy(self) -> "SplitSpec":
        if self.strategy == "prompt_fixed":
            if (
                self.validation_fraction is None
                or not 0.0 < self.validation_fraction < 1.0
            ):
                raise ValueError(
                    "prompt_fixed split requires validation_fraction strictly between 0 and 1"
                )
        elif self.validation_fraction is not None:
            raise ValueError("preserve split must not set validation_fraction")
        return self


class BuildSpec(StrictModel):
    spec_version: Literal[1, 2, 3, 4]
    dataset: DatasetIdentity
    policy: PolicySpec
    sources: tuple[SourceSpec, ...]
    selection: SelectionSpec
    split: SplitSpec

    @model_validator(mode="after")
    def validate_sources(self) -> "BuildSpec":
        if not self.sources:
            raise ValueError("At least one dataset source is required")
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("Dataset source IDs must be unique")
        if any(source.partition == "test" for source in self.sources):
            raise ValueError("SFT dataset builds must not include test partitions")
        partitions = {source.partition for source in self.sources}
        if self.split.strategy == "prompt_fixed" and partitions != {"train"}:
            raise ValueError("prompt_fixed split accepts only train sources")
        if self.split.strategy == "preserve" and not {"train", "validation"}.issubset(
            partitions
        ):
            raise ValueError("preserve split requires train and validation sources")
        if self.spec_version == 1:
            if self.policy.subagent_types:
                raise ValueError(
                    "spec_version 1 does not support policy subagent types"
                )
            if self.selection.policy != "exact_reward":
                raise ValueError("spec_version 1 requires exact_reward selection")
            if any(
                source.trace_format != "native"
                or source.subagent_type_aliases
                or source.sampling is not None
                or source.expected_native_rollouts is not None
                or source.expected_candidates is not None
                or source.require_completed_run
                or source.selection is not None
                for source in self.sources
            ):
                raise ValueError("spec_version 1 does not support v2 source options")
        else:
            # Without policy.subagent_types, records keep their native tool schemas.
            allowed_ids = {subagent.id for subagent in self.policy.subagent_types}
            for source in self.sources:
                unknown_targets = sorted(
                    set(source.subagent_type_aliases.values()) - allowed_ids
                )
                if unknown_targets:
                    raise ValueError(
                        f"Source {source.id!r} aliases unknown canonical subagent "
                        "types: " + ", ".join(unknown_targets)
                    )
                if source.expected_native_rollouts is None:
                    raise ValueError(
                        f"spec_version >=2 source {source.id!r} must pin "
                        "expected_native_rollouts"
                    )
                if source.expected_candidates is None:
                    raise ValueError(
                        f"spec_version >=2 source {source.id!r} must pin "
                        "expected_candidates"
                    )
        for source in self.sources:
            if self.spec_version >= 4 and (
                source.snapshot is None or source.path is not None
            ):
                raise ValueError(
                    f"spec_version 4 source {source.id!r} must name a snapshot "
                    "and no path"
                )
            if self.spec_version < 4 and source.snapshot is not None:
                raise ValueError("source snapshots require spec_version 4")
        if self.spec_version < 3 and (
            self.selection.policy
            in {
                "toolathlon_pass_or_quality",
                "toolathlon_pass_or_quality_inclusive",
                "collector_qualifies",
            }
            or any(source.selection is not None for source in self.sources)
        ):
            raise ValueError("source-specific selection requires spec_version 3")
        for source in self.sources:
            effective_policy = (
                source.selection.policy
                if source.selection is not None
                else self.selection.policy
            )
            if (
                effective_policy
                in {
                    "toolathlon_pass_or_quality",
                    "toolathlon_pass_or_quality_inclusive",
                }
                and source.adapter != "toolathlon_gym"
            ):
                raise ValueError(
                    "toolathlon_pass_or_quality is only valid for Toolathlon sources"
                )
            if (effective_policy == "collector_qualifies") != (
                source.trace_format in COLLECTION_TRACE_FORMATS
            ):
                raise ValueError(
                    f"Source {source.id!r}: collection sources "
                    f"({', '.join(sorted(COLLECTION_TRACE_FORMATS))}), and only "
                    "they, select with collector_qualifies"
                )
        return self


class CanonicalSource(StrictModel):
    adapter: str
    adapter_version: int
    source_id: str
    benchmark: str
    environment: str
    partition: Literal["train", "validation"]
    teacher: str
    task_id: str
    rollout_id: str


class CanonicalOutcome(StrictModel):
    success: bool
    reward: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class CanonicalRollout(StrictModel):
    schema_version: Literal[1] = 1
    id: str
    group_id: str
    messages: list[JsonObject]
    tools: list[JsonObject]
    source: CanonicalSource
    outcome: CanonicalOutcome
    attributes: JsonObject = Field(default_factory=dict)


class TraceValidationError(ValueError):
    """A reason-coded error local to one native rollout."""

    def __init__(self, reason: str, message: str) -> None:
        if reason not in EXCLUSION_REASONS or reason in {
            "excluded_reward",
            "excluded_prompt_teacher_cap",
        }:
            raise ValueError(f"Invalid trace-exclusion reason: {reason}")
        super().__init__(message)
        self.reason = reason


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: Any, description: str, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceValidationError(
            reason,
            f"Expected {description} to be an object, got {type(value).__name__}.",
        )
    return value


def reject_legacy_tool_name(name: Any, description: str) -> None:
    """Exclude traces of the retired spawn_subagent/wait Decomposer core."""
    if name in LEGACY_DECOMPOSER_TOOL_NAMES:
        raise TraceValidationError(
            "excluded_legacy_tool_interface",
            f"{description} uses the retired {name!r} tool; SFT preparation "
            "accepts only new/fork/run/wait Decomposer trajectories.",
        )


def normalize_response_tools(tools: Any) -> list[JsonObject]:
    """Normalize Responses API function tools into Transformers chat format."""
    if not isinstance(tools, list):
        raise TraceValidationError(
            "excluded_invalid_tool_schema", "response.tools must be a list."
        )
    normalized: list[JsonObject] = []
    names: list[str] = []
    for index, raw_tool in enumerate(tools):
        tool = require_mapping(
            raw_tool, f"response.tools[{index}]", "excluded_invalid_tool_schema"
        )
        if tool.get("type") != "function":
            raise TraceValidationError(
                "excluded_invalid_tool_schema",
                f"Unsupported tool type at response.tools[{index}]: {tool.get('type')!r}.",
            )
        name = tool.get("name")
        description = tool.get("description")
        parameters = tool.get("parameters")
        if not isinstance(name, str) or not name:
            raise TraceValidationError(
                "excluded_invalid_tool_schema", f"Tool {index} has no name."
            )
        reject_legacy_tool_name(name, f"response.tools[{index}]")
        if not isinstance(description, str) or not isinstance(parameters, Mapping):
            raise TraceValidationError(
                "excluded_invalid_tool_schema", f"Tool {name!r} has an invalid schema."
            )
        names.append(name)
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": dict(parameters),
                },
            }
        )
    # Apply the same parameter and tool-set checks as embedded chat schemas.
    return validate_chat_tools(normalized)


def validate_chat_tools(tools: Any) -> list[JsonObject]:
    """Validate canonical chat function schemas embedded in a native trace."""
    if not isinstance(tools, list):
        raise TraceValidationError(
            "excluded_invalid_tool_schema", "trace.tools must be a list."
        )
    validated: list[JsonObject] = []
    names: list[str] = []
    for index, raw_tool in enumerate(tools):
        tool = require_mapping(
            raw_tool, f"trace.tools[{index}]", "excluded_invalid_tool_schema"
        )
        function = require_mapping(
            tool.get("function"),
            f"trace.tools[{index}].function",
            "excluded_invalid_tool_schema",
        )
        name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters")
        reject_legacy_tool_name(name, f"trace.tools[{index}]")
        if (
            tool.get("type") != "function"
            or not isinstance(name, str)
            or not name
            or not isinstance(description, str)
            or not description
            or not isinstance(parameters, Mapping)
        ):
            raise TraceValidationError(
                "excluded_invalid_tool_schema",
                f"trace.tools[{index}] is not a valid function schema.",
            )
        properties = parameters.get("properties")
        required = parameters.get("required", [])
        if parameters.get("type") != "object" or not isinstance(properties, Mapping):
            raise TraceValidationError(
                "excluded_invalid_tool_schema",
                f"Tool {name!r} must have object parameters.",
            )
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise TraceValidationError(
                "excluded_invalid_tool_schema",
                f"Tool {name!r} has an invalid required-parameter list.",
            )
        expected = DECOMPOSER_TOOL_PARAMETERS.get(name)
        if expected is not None:
            if set(properties) != set(expected) or set(required) != set(expected):
                raise TraceValidationError(
                    "excluded_invalid_tool_schema",
                    f"{name} must take no arguments."
                    if not expected
                    else f"{name} must require exactly: {', '.join(expected)}.",
                )
            if any(
                not isinstance(properties[item], Mapping)
                or properties[item].get("type") != "string"
                for item in expected
            ):
                raise TraceValidationError(
                    "excluded_invalid_tool_schema",
                    f"{name} parameters must be strings.",
                )
        names.append(name)
        validated.append(dict(tool))
    if len(names) != len(DECOMPOSER_TOOL_NAMES) or set(names) != DECOMPOSER_TOOL_NAMES:
        raise TraceValidationError(
            "excluded_invalid_tool_schema",
            "Exposed tools must be exactly one each of new, fork, run, and wait.",
        )
    return validated


def normalize_subagent_type_ids(
    messages: list[JsonObject],
    *,
    allowed_ids: frozenset[str],
    aliases: Mapping[str, str],
) -> int:
    """Normalize ``new``-call ``agent_type_id`` values to one canonical policy interface.

    Only ``new`` names a subagent type; ``fork`` and ``run`` address existing
    subagents by ``agent_id`` and inherit their type.
    """
    if not allowed_ids:
        return 0
    normalized = 0
    for message_index, message in enumerate(messages):
        for raw_call in message.get("tool_calls") or []:
            call = require_mapping(
                raw_call,
                f"assistant message {message_index} tool call",
                "excluded_invalid_tool_calls",
            )
            function = require_mapping(
                call.get("function"),
                "tool-call function",
                "excluded_invalid_tool_calls",
            )
            if function.get("name") != "new":
                continue
            arguments = require_mapping(
                function.get("arguments"),
                "new arguments",
                "excluded_invalid_tool_calls",
            )
            raw_id = arguments.get("agent_type_id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise TraceValidationError(
                    "excluded_invalid_tool_calls",
                    "new has no valid agent_type_id.",
                )
            canonical_id = aliases.get(raw_id, raw_id)
            if canonical_id not in allowed_ids:
                raise TraceValidationError(
                    "excluded_invalid_tool_calls",
                    f"Unknown agent_type_id {raw_id!r}.",
                )
            if canonical_id != raw_id:
                if not isinstance(arguments, dict):
                    raise TraceValidationError(
                        "excluded_invalid_tool_calls",
                        "new arguments must be mutable JSON objects.",
                    )
                arguments["agent_type_id"] = canonical_id
                normalized += 1
    return normalized


def _validate_call_arguments(
    name: str,
    arguments: Mapping[str, Any],
    subagent_type_ids: frozenset[str],
) -> None:
    if name == "new":
        type_id = arguments.get("agent_type_id")
        if (
            set(arguments) != {"agent_type_id"}
            or not isinstance(type_id, str)
            or not type_id.strip()
        ):
            raise TraceValidationError(
                "excluded_invalid_tool_calls",
                "new requires exactly one non-empty agent_type_id string.",
            )
        if subagent_type_ids and type_id not in subagent_type_ids:
            raise TraceValidationError(
                "excluded_invalid_tool_calls",
                f"Unknown agent_type_id {type_id!r}.",
            )
    elif name == "fork":
        if set(arguments) != {"agent_id"} or not isinstance(
            arguments["agent_id"], str
        ):
            raise TraceValidationError(
                "excluded_invalid_tool_calls",
                "fork requires exactly one agent_id string.",
            )
    elif name == "run":
        prompt = arguments.get("prompt")
        if (
            set(arguments) != {"agent_id", "prompt"}
            or not isinstance(arguments["agent_id"], str)
            or not isinstance(prompt, str)
            or not prompt.strip()
        ):
            raise TraceValidationError(
                "excluded_invalid_tool_calls",
                "run requires an agent_id string and a non-empty prompt string.",
            )
    elif arguments:
        raise TraceValidationError(
            "excluded_invalid_tool_calls", "wait arguments must be empty."
        )


def validate_decomposer_messages(
    messages: list[JsonObject],
    *,
    subagent_type_ids: frozenset[str] = frozenset(),
    allow_core_errors: bool = False,
) -> None:
    """Validate the benchmark-neutral Decomposer tool-calling trajectory.

    ``subagent_type_ids``, when non-empty, lists the subagent types ``new`` may
    create; otherwise any non-empty type ID is accepted.

    ``allow_core_errors`` keeps the mistakes the core answered and the manager
    recovered from: calls of any tool name with any arguments, which the core
    rejects with an error result or runs ignoring extra arguments, and the user
    messages the core injects (``CORE_USER_MESSAGES``), including the empty
    answer that precedes ``EMPTY_RESPONSE_ERROR``. The structure is still
    checked: one task, every call answered, and a final text answer.
    """
    if len(messages) < 3 or messages[0].get("role") != "system":
        raise TraceValidationError(
            "excluded_invalid_messages",
            "A canonical trace must start with a system message.",
        )
    if messages[1].get("role") != "user":
        raise TraceValidationError(
            "excluded_invalid_messages",
            "The system message must be followed by one user message.",
        )
    final = messages[-1]
    if (
        final.get("role") != "assistant"
        or final.get("tool_calls")
        or not isinstance(final.get("content"), str)
        or not final["content"].strip()
    ):
        raise TraceValidationError(
            "excluded_empty_training_target",
            "The final assistant message must contain text and no tool calls.",
        )

    pending: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    for index, message in enumerate(messages[1:], start=1):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            raise TraceValidationError(
                "excluded_invalid_messages",
                f"Message {index} content must be a string.",
            )
        if role == "user":
            injected = allow_core_errors and content in CORE_USER_MESSAGES
            if (index != 1 and not injected) or pending:
                raise TraceValidationError(
                    "excluded_invalid_messages",
                    "User messages may only appear after system.",
                )
            continue
        if role == "assistant":
            if pending:
                raise TraceValidationError(
                    "excluded_invalid_tool_calls",
                    f"Assistant message arrived before tool results for {sorted(pending)}.",
                )
            raw_calls = message.get("tool_calls") or []
            if not isinstance(raw_calls, list):
                raise TraceValidationError(
                    "excluded_invalid_tool_calls",
                    f"Assistant message {index} has invalid calls.",
                )
            answered_empty = (
                allow_core_errors
                and index + 1 < len(messages)
                and messages[index + 1].get("role") == "user"
                and messages[index + 1].get("content") == EMPTY_RESPONSE_ERROR
            )
            if not content.strip() and not raw_calls and not answered_empty:
                raise TraceValidationError(
                    "excluded_empty_training_target",
                    f"Assistant message {index} has neither text nor a tool call.",
                )
            for raw_call in raw_calls:
                call = require_mapping(
                    raw_call,
                    f"assistant message {index} tool call",
                    "excluded_invalid_tool_calls",
                )
                call_id = call.get("id")
                function = require_mapping(
                    call.get("function"),
                    "tool-call function",
                    "excluded_invalid_tool_calls",
                )
                name = function.get("name")
                arguments = function.get("arguments")
                reject_legacy_tool_name(name, f"Assistant message {index}")
                known_name = (
                    isinstance(name, str) and bool(name)
                    if allow_core_errors
                    else name in DECOMPOSER_TOOL_NAMES
                )
                if (
                    call.get("type") != "function"
                    or not isinstance(call_id, str)
                    or not call_id
                    or not known_name
                    or not isinstance(arguments, Mapping)
                ):
                    raise TraceValidationError(
                        "excluded_invalid_tool_calls",
                        f"Assistant message {index} has an invalid call.",
                    )
                if not allow_core_errors:
                    _validate_call_arguments(str(name), arguments, subagent_type_ids)
                if call_id in seen_call_ids:
                    raise TraceValidationError(
                        "excluded_invalid_tool_calls",
                        f"Duplicate tool-call ID {call_id!r}.",
                    )
                seen_call_ids.add(call_id)
                pending[call_id] = str(name)
            continue
        if role != "tool":
            raise TraceValidationError(
                "excluded_invalid_messages",
                f"Unsupported canonical role {role!r} at {index}.",
            )
        call_id = message.get("tool_call_id")
        name = message.get("name")
        if (
            not isinstance(call_id, str)
            or call_id not in pending
            or pending[call_id] != name
        ):
            raise TraceValidationError(
                "excluded_invalid_tool_calls",
                f"Tool message references unknown or mismatched call {call_id!r}.",
            )
        del pending[call_id]
    if pending:
        raise TraceValidationError(
            "excluded_invalid_tool_calls",
            f"Missing tool results for {sorted(pending)}.",
        )
