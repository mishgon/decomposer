#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path
from typing import Any


def _json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content)

    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(item.get("text") or item.get("content") or json.dumps(item, ensure_ascii=False))
        else:
            parts.append(str(item))
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _text_block(text: str) -> list[str]:
    text = text.strip()
    longest_fence = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest_fence + 1)
    return [f"{fence}text", text, fence]


def _render_rollout(row: dict[str, Any], number: int) -> list[str]:
    response = row.get("response") or {}
    usage = response.get("usage") or {}
    output = response.get("output") or []

    lines = [f"# Rollout {number}", "", "| Field | Value |", "| --- | --- |"]
    fields = [
        ("Model", response.get("model")),
        ("Status", response.get("status")),
        ("Reward", row.get("reward")),
        ("Expected answer", row.get("expected_answer")),
        ("Extracted answer", row.get("extracted_answer")),
        ("Input tokens", usage.get("input_tokens")),
        ("Output tokens", usage.get("output_tokens")),
        ("Total tokens", usage.get("total_tokens")),
    ]
    lines.extend(f"| {name} | {value} |" for name, value in fields if value is not None)

    lines.extend(["", "## Input", ""])
    request = row.get("responses_create_params") or {}
    for message in request.get("input") or []:
        role = str(message.get("role", "message")).replace("_", " ").title()
        lines.extend([f"### {role}", ""])
        lines.extend(_text_block(_content_text(message.get("content", ""))))
        lines.append("")

    calls: dict[str, tuple[str, Any]] = {}
    run_types: dict[str, str] = {}
    for item in output:
        if item.get("type") == "function_call":
            calls[item.get("call_id") or item.get("id")] = (item.get("name", "tool"), _json(item.get("arguments", {})))
        elif item.get("type") == "function_call_output":
            call_id = item.get("call_id")
            name, arguments = calls.get(call_id, ("", {}))
            result = _json(item.get("output"))
            if name == "spawn_subagent" and isinstance(arguments, dict) and isinstance(result, dict):
                run_id = result.get("subagent_run_id")
                if run_id:
                    run_types[run_id] = arguments.get("subagent_type_id", "unknown subagent")

    lines.extend(["## Trace", ""])
    step = 0
    for item in output:
        item_type = item.get("type")

        if item_type == "reasoning":
            step += 1
            lines.extend([f"### {step}. Decomposer reasoning summary", ""])
            for summary in item.get("summary") or []:
                text = summary.get("text", "") if isinstance(summary, dict) else str(summary)
                if text:
                    lines.extend(_text_block(text))
                    lines.append("")

        elif item_type == "function_call":
            step += 1
            name = item.get("name", "tool")
            arguments = _json(item.get("arguments", {}))
            if name == "spawn_subagent" and isinstance(arguments, dict):
                subagent_type = arguments.get("subagent_type_id", "unknown")
                lines.extend([f"### {step}. Spawn `{subagent_type}`", "", "**Prompt**", ""])
                lines.extend(_text_block(str(arguments.get("prompt", ""))))
                lines.append("")
            elif name == "wait":
                lines.extend([f"### {step}. Wait", ""])
            else:
                lines.extend([f"### {step}. Call `{name}`", "", "```json", json.dumps(arguments, indent=2, ensure_ascii=False), "```", ""])

        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            name, _ = calls.get(call_id, ("", {}))
            result = _json(item.get("output"))
            if name == "wait" and isinstance(result, list):
                for report in result:
                    run_id = report.get("subagent_run_id", "unknown")
                    subagent_type = run_types.get(run_id, "unknown subagent")
                    lines.extend(
                        [
                            f"#### Report from `{subagent_type}`",
                            "",
                            f"Status: `{report.get('status', 'unknown')}` · Run: `{run_id}`",
                            "",
                        ]
                    )
                    lines.extend(_text_block(str(report.get("content", ""))))
                    lines.append("")
            elif name not in {"spawn_subagent", "wait"}:
                lines.extend(["**Tool output**", "", "```json", json.dumps(result, indent=2, ensure_ascii=False), "```", ""])

        elif item_type == "message":
            step += 1
            lines.extend([f"### {step}. Decomposer final response", ""])
            lines.extend(_text_block(_content_text(item.get("content", ""))))
            lines.append("")

    return lines


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} INPUT.jsonl [OUTPUT.md]")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else input_path.with_suffix(".md")
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]

    rendered: list[str] = []
    for number, row in enumerate(rows, start=1):
        if rendered:
            rendered.extend(["", "---", ""])
        rendered.extend(_render_rollout(row, number))

    output_path.write_text("\n".join(rendered).rstrip() + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
