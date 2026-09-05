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

JsonObject = dict[str, Any]
InvalidPolicy = Literal["exclude", "error"]
SourcePartition = Literal["train", "validation", "test"]
SelectionPolicy = Literal[
    "exact_reward",
    "all_rewards",
    "toolathlon_pass_or_quality",
    "toolathlon_pass_or_quality_inclusive",
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
    "excluded_multiple_tool_calls",
    "excluded_invalid_messages",
    "excluded_invalid_metadata",
    "excluded_prompt_teacher_cap",
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


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
        # Missing and legacy ``decomposer_default`` both preserve the historical
        # student-prompt behavior.
        return self.system_prompt_profile or "student"


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


class Gaia2SourceSpec(StrictModel):
    split_manifest: Path
    scenario_partition: Literal["train"] = "train"
    expected_scenarios: int
    logical_rollout_numbers: tuple[int, ...]

    @field_validator("expected_scenarios")
    @classmethod
    def validate_expected_scenarios(cls, value: int) -> int:
        if isinstance(value, bool) or value <= 0:
            raise ValueError("gaia2.expected_scenarios must be a positive integer")
        return value

    @field_validator("logical_rollout_numbers")
    @classmethod
    def validate_logical_rollout_numbers(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        if (
            not value
            or any(isinstance(item, bool) or item <= 0 for item in value)
            or len(value) != len(set(value))
            or tuple(sorted(value)) != value
        ):
            raise ValueError(
                "gaia2.logical_rollout_numbers must be unique, positive, and sorted"
            )
        return value


class SourceSpec(StrictModel):
    id: str
    adapter: Literal["gaia2", "nemo_gym", "toolathlon_gym"]
    path: Path | None = None
    benchmark: str
    environment: str
    partition: SourcePartition
    teacher: str
    trace_format: Literal[
        "native",
        "toolathlon_legacy_unversioned",
        "gaia2_evaluation_v1",
        "gaia2_trace_manifest_v1",
    ] = "native"
    subagent_type_aliases: dict[str, str] = Field(default_factory=dict)
    sampling: SourceSamplingSpec | None = None
    selection: SourceSelectionSpec | None = None
    expected_native_rollouts: int | None = None
    expected_candidates: int | None = None
    require_completed_run: bool = False
    gaia2: Gaia2SourceSpec | None = None

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
        gaia2_formats = {"gaia2_evaluation_v1", "gaia2_trace_manifest_v1"}
        if (
            self.trace_format == "toolathlon_legacy_unversioned"
            and self.adapter != "toolathlon_gym"
        ):
            raise ValueError(
                "toolathlon_legacy_unversioned is only valid for Toolathlon sources"
            )
        if (self.trace_format in gaia2_formats) != (self.adapter == "gaia2"):
            raise ValueError("GAIA2 trace formats are only valid for GAIA2 sources")
        if (self.gaia2 is not None) != (self.adapter == "gaia2"):
            raise ValueError("gaia2 options are required only for GAIA2 sources")
        if self.sampling is not None and self.adapter != "nemo_gym":
            raise ValueError("source task sampling is only supported for NeMo Gym")
        if self.require_completed_run and self.adapter not in {
            "gaia2",
            "toolathlon_gym",
        }:
            raise ValueError(
                "require_completed_run is only valid for GAIA2 or Toolathlon"
            )
        if self.adapter == "gaia2" and not self.require_completed_run:
            raise ValueError("GAIA2 SFT sources must require a completed run")
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
        if self.gaia2 is not None and self.expected_candidates is not None:
            expected = self.gaia2.expected_scenarios * len(
                self.gaia2.logical_rollout_numbers
            )
            if self.expected_candidates != expected:
                raise ValueError("expected_candidates must match the GAIA2 task layout")
        return self


class SplitSpec(StrictModel):
    strategy: Literal["pinned", "prompt_fixed", "preserve"]
    seed: int = 42
    validation_fraction: float | None = None
    manifest: Path | None = None

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
            if self.manifest is not None:
                raise ValueError("prompt_fixed split must not set manifest")
        elif self.strategy == "pinned":
            if self.validation_fraction is not None or self.manifest is None:
                raise ValueError(
                    "pinned split requires manifest and no validation_fraction"
                )
        elif self.validation_fraction is not None or self.manifest is not None:
            raise ValueError(
                "preserve split must not set validation_fraction or manifest"
            )
        return self


class TokenizationSpec(StrictModel):
    profile: Literal[
        "gemma4_sft_non_thinking",
        "qwen35_sft_non_thinking",
    ]
    tokenizer: str
    revision: str = "main"
    max_tokens: int = 32768
    trust_remote_code: bool = False

    @field_validator("tokenizer", "revision")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tokenization string fields must be non-empty")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        if isinstance(value, bool) or not 0 < value <= 32768:
            raise ValueError("tokenization.max_tokens must be between 1 and 32768")
        return value


class BuildSpec(StrictModel):
    spec_version: Literal[1, 2, 3]
    dataset: DatasetIdentity
    policy: PolicySpec
    sources: tuple[SourceSpec, ...]
    selection: SelectionSpec
    split: SplitSpec
    tokenization: TokenizationSpec | None = None

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
        if self.split.strategy in {"pinned", "prompt_fixed"} and partitions != {
            "train"
        }:
            raise ValueError(f"{self.split.strategy} split accepts only train sources")
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
                or source.gaia2 is not None
                for source in self.sources
            ):
                raise ValueError("spec_version 1 does not support v2 source options")
        else:
            if not self.policy.subagent_types:
                raise ValueError("spec_version >=2 requires policy.subagent_types")
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
        if self.spec_version < 3 and (
            self.selection.policy
            in {
                "toolathlon_pass_or_quality",
                "toolathlon_pass_or_quality_inclusive",
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
    if len(names) != 2 or set(names) != {"spawn_subagent", "wait"}:
        raise TraceValidationError(
            "excluded_invalid_tool_schema",
            "Exposed tools must be exactly one spawn_subagent and one wait tool.",
        )
    return normalized


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
        if name == "spawn_subagent":
            expected = {"subagent_type_id", "prompt"}
            if set(properties) != expected or set(required) != expected:
                raise TraceValidationError(
                    "excluded_invalid_tool_schema",
                    "spawn_subagent must require subagent_type_id and prompt.",
                )
            if any(
                not isinstance(properties[item], Mapping)
                or properties[item].get("type") != "string"
                for item in expected
            ):
                raise TraceValidationError(
                    "excluded_invalid_tool_schema",
                    "spawn_subagent parameters must be strings.",
                )
        elif name == "wait":
            if properties or required:
                raise TraceValidationError(
                    "excluded_invalid_tool_schema", "wait must take no arguments."
                )
        names.append(name)
        validated.append(dict(tool))
    if len(names) != 2 or set(names) != {"spawn_subagent", "wait"}:
        raise TraceValidationError(
            "excluded_invalid_tool_schema",
            "Exposed tools must be exactly one spawn_subagent and one wait tool.",
        )
    return validated


def normalize_subagent_type_ids(
    messages: list[JsonObject],
    *,
    allowed_ids: frozenset[str],
    aliases: Mapping[str, str],
) -> int:
    """Normalize spawn-call subagent IDs to one canonical policy interface."""
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
            if function.get("name") != "spawn_subagent":
                continue
            arguments = require_mapping(
                function.get("arguments"),
                "spawn_subagent arguments",
                "excluded_invalid_tool_calls",
            )
            raw_id = arguments.get("subagent_type_id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise TraceValidationError(
                    "excluded_invalid_tool_calls",
                    "spawn_subagent has no valid subagent_type_id.",
                )
            canonical_id = aliases.get(raw_id, raw_id)
            if canonical_id not in allowed_ids:
                raise TraceValidationError(
                    "excluded_invalid_tool_calls",
                    f"Unknown subagent_type_id {raw_id!r}.",
                )
            if canonical_id != raw_id:
                if not isinstance(arguments, dict):
                    raise TraceValidationError(
                        "excluded_invalid_tool_calls",
                        "spawn_subagent arguments must be mutable JSON objects.",
                    )
                arguments["subagent_type_id"] = canonical_id
                normalized += 1
    return normalized


def sequentialize_parallel_spawn_calls(
    messages: list[JsonObject],
) -> tuple[list[JsonObject], int, int, int, int]:
    """Convert parallel spawn batches into single-call assistant/tool turns.

    Decomposer may emit several asynchronous ``spawn_subagent`` calls in one
    assistant message. The canonical SFT format intentionally keeps one tool
    call per assistant message, so each parallel batch is paired with its tool
    results by call ID and emitted in the teacher's original call order.

    Shared assistant content and teacher reasoning belong to the original
    completion and are retained only on the first sequentialized turn.

    A ``wait`` inside such a batch is dropped along with the tool message
    answering it. The harness refuses to run those calls -- it replies "A `wait`
    call must be the only tool call in the message. This `wait` call was not
    executed." -- so removing them reproduces the trajectory the environment
    actually saw rather than editing away a real action. When a batch holds
    nothing but waits the whole turn goes, text and all, because what remains is
    usually a summary truncated mid-sentence by the token limit that produced the
    stray calls in the first place.

    A lone ``wait`` is a valid, executed call and passes through untouched: only
    multi-call messages are rewritten, matching the harness's own rule.

    Returns the rewritten messages, the number of batches sequentialized, the
    number of spawn calls they contained, and the number of dropped wait calls
    and dropped turns.
    """
    normalized: list[JsonObject] = []
    normalized_messages = 0
    normalized_calls = 0
    dropped_wait_calls = 0
    dropped_turns = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        raw_calls = message.get("tool_calls") or []
        if message.get("role") != "assistant" or not isinstance(raw_calls, list):
            normalized.append(message)
            index += 1
            continue
        if len(raw_calls) <= 1:
            normalized.append(message)
            index += 1
            continue

        calls: list[Mapping[str, Any]] = []
        call_ids: list[str] = []
        wait_ids: list[str] = []
        for raw_call in raw_calls:
            call = require_mapping(
                raw_call,
                f"assistant message {index} tool call",
                "excluded_invalid_tool_calls",
            )
            function = require_mapping(
                call.get("function"),
                f"assistant message {index} tool-call function",
                "excluded_invalid_tool_calls",
            )
            call_id = call.get("id")
            name = function.get("name")
            if (
                call.get("type") != "function"
                or name not in {"spawn_subagent", "wait"}
                or not isinstance(call_id, str)
                or not call_id
            ):
                raise TraceValidationError(
                    "excluded_invalid_tool_calls",
                    f"Assistant message {index} shares an invalid tool call.",
                )
            if name == "wait":
                wait_ids.append(call_id)
                continue
            calls.append(call)
            call_ids.append(call_id)
        if len(call_ids) != len(set(call_ids)):
            raise TraceValidationError(
                "excluded_invalid_tool_calls",
                f"Assistant message {index} contains duplicate tool-call IDs.",
            )

        result_end = index + 1
        while result_end < len(messages) and messages[result_end].get("role") == "tool":
            result_end += 1
        results = messages[index + 1 : result_end]
        results_by_id: dict[str, JsonObject] = {}
        for result in results:
            result_id = result.get("tool_call_id")
            if not isinstance(result_id, str) or result_id in results_by_id:
                raise TraceValidationError(
                    "excluded_invalid_tool_calls",
                    f"Parallel assistant message {index} has malformed tool results.",
                )
            results_by_id[result_id] = result
        if not all(call_id in results_by_id for call_id in call_ids):
            raise TraceValidationError(
                "excluded_invalid_tool_calls",
                f"Parallel assistant message {index} must be followed by exactly one "
                "matching tool result for every spawn call.",
            )

        dropped_wait_calls += len(wait_ids)
        if not calls:
            # Nothing the environment executed survives, so the turn carries no
            # supervision worth keeping.
            dropped_turns += 1
            index = result_end
            continue

        for call_index, (call, call_id) in enumerate(zip(calls, call_ids)):
            split_message = dict(message)
            split_message["tool_calls"] = [dict(call)]
            if call_index:
                split_message["content"] = ""
                split_message.pop("teacher_reasoning", None)
            normalized.extend((split_message, results_by_id[call_id]))

        normalized_messages += 1
        normalized_calls += len(calls)
        index = result_end

    return (
        normalized,
        normalized_messages,
        normalized_calls,
        dropped_wait_calls,
        dropped_turns,
    )


def validate_decomposer_messages(messages: list[JsonObject]) -> None:
    """Validate the benchmark-neutral Decomposer tool-calling trajectory."""
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
            if index != 1 or pending:
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
            if len(raw_calls) > 1:
                raise TraceValidationError(
                    "excluded_multiple_tool_calls",
                    f"Assistant message {index} contains {len(raw_calls)} tool calls.",
                )
            if not content.strip() and not raw_calls:
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
                if (
                    call.get("type") != "function"
                    or not isinstance(call_id, str)
                    or not call_id
                    or name not in {"spawn_subagent", "wait"}
                    or not isinstance(arguments, Mapping)
                ):
                    raise TraceValidationError(
                        "excluded_invalid_tool_calls",
                        f"Assistant message {index} has an invalid call.",
                    )
                arguments = dict(arguments)
                if name == "spawn_subagent":
                    if set(arguments) != {"subagent_type_id", "prompt"} or not all(
                        isinstance(arguments[key], str) and arguments[key].strip()
                        for key in ("subagent_type_id", "prompt")
                    ):
                        raise TraceValidationError(
                            "excluded_invalid_tool_calls",
                            "spawn_subagent requires non-empty subagent_type_id "
                            "and prompt strings.",
                        )
                elif arguments:
                    raise TraceValidationError(
                        "excluded_invalid_tool_calls", "wait arguments must be empty."
                    )
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
