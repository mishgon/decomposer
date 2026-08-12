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

CANONICAL_SCHEMA_VERSION = 1
MANIFEST_FORMAT_VERSION = 3

EXCLUSION_REASONS = (
    "excluded_reward",
    "excluded_invalid_json",
    "excluded_invalid_reward",
    "excluded_invalid_indices",
    "excluded_invalid_agent_ref",
    "excluded_missing_materialized_input",
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


class PolicySpec(StrictModel):
    id: str
    system_prompt: Literal["decomposer_default"] = "decomposer_default"

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("policy.id must be a lowercase identifier")
        return value


class SourceSpec(StrictModel):
    id: str
    adapter: Literal["nemo_gym"]
    path: Path
    benchmark: str
    environment: str
    partition: SourcePartition
    teacher: str

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


class SelectionSpec(StrictModel):
    policy: Literal["exact_reward"] = "exact_reward"
    success_reward: float = 1.0
    invalid_policy: InvalidPolicy = "exclude"
    max_traces_per_prompt_per_teacher: int | None = None

    @field_validator("success_reward")
    @classmethod
    def validate_reward(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("selection.success_reward must be finite")
        return value

    @field_validator("max_traces_per_prompt_per_teacher")
    @classmethod
    def validate_cap(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError(
                "selection.max_traces_per_prompt_per_teacher must be positive"
            )
        return value


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
    spec_version: Literal[1]
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
                            "spawn_subagent requires non-empty subagent_type_id and prompt strings.",
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
