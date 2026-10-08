"""Convert a Decomposer manager's LangChain messages to canonical chat messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schema import (
    JsonObject,
    TraceValidationError,
    reject_legacy_tool_name,
    require_mapping,
    validate_decomposer_messages,
)


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
    reject_legacy_tool_name(name, f"Assistant message {message_index}")
    if (
        not isinstance(call_id, str)
        or not call_id
        or not isinstance(name, str)
        or not name
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
            f"messages[{index}].content must be a string.",
        )
    if message_type == "human":
        return {"role": "user", "content": content}
    if message_type == "tool":
        call_id = message.get("tool_call_id")
        name = message.get("name")
        reject_legacy_tool_name(name, f"Tool message {index}")
        if (
            not isinstance(call_id, str)
            or not call_id
            or not isinstance(name, str)
            or not name
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
            f"Unsupported message type {message_type!r} at index {index}.",
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


def convert_langgraph_messages(messages: Any, system_prompt: str) -> list[JsonObject]:
    """Prepend the system prompt and keep the mistakes the core answered.

    ``messages`` is the manager's LangGraph state: LangChain message dumps that
    start with the task and end with the final answer.
    """
    if not isinstance(messages, list) or not messages:
        raise TraceValidationError(
            "excluded_missing_final_state",
            "The manager messages must be a non-empty list.",
        )
    if not all(isinstance(message, Mapping) for message in messages):
        raise TraceValidationError(
            "excluded_invalid_messages", "Every manager message must be an object."
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
    validate_decomposer_messages(converted, allow_core_errors=True)
    return converted
