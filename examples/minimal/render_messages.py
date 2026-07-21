import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _tool_output(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def render_decomposer_messages(messages: Sequence[BaseMessage]) -> str:
    tool_names = {
        tool_call["id"]: tool_call["name"]
        for message in messages
        for tool_call in (getattr(message, "tool_calls", None) or [])
    }

    lines = ["# Decomposer messages", ""]
    for index, message in enumerate(messages, start=1):
        if message.type == "human":
            heading = "User"
        elif message.type == "ai":
            heading = "Decomposer"
        elif message.type == "tool":
            tool_name = tool_names.get(getattr(message, "tool_call_id", ""), "tool")
            heading = f"Tool result: `{tool_name}`"
        else:
            heading = message.type.replace("_", " ").title()
        lines.extend([f"## {index}. {heading}", ""])

        if message.content:
            if message.type == "tool":
                lines.extend(["```json", _json(_tool_output(message.content)), "```", ""])
            else:
                lines.extend([str(message.content), ""])

        for tool_call in getattr(message, "tool_calls", None) or []:
            lines.extend(
                [
                    f"### Call `{tool_call['name']}`",
                    "",
                    f"ID: `{tool_call['id']}`",
                    "",
                    "```json",
                    _json(tool_call.get("args", {})),
                    "```",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"
