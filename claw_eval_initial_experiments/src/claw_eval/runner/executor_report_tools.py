"""Executor-local tool used to submit structured coordinator reports."""

from __future__ import annotations

from typing import Any

from ..models.tool import ToolSpec

SUBMIT_EXECUTOR_REPORT_NAME = "submit_report"

SUBMIT_EXECUTOR_REPORT_TOOL = ToolSpec(
    name=SUBMIT_EXECUTOR_REPORT_NAME,
    description=(
        "Submit the final compact report for the coordinator after environment "
        "tool use is complete. Call this tool alone, after observing needed tool results."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "what_did": {
                "type": "string",
                "description": "Brief description of the actions performed.",
            },
            "key_findings": {
                "type": "string",
                "description": "Concrete findings, outputs, or answer fragments discovered.",
            },
            "tool_results": {
                "type": "string",
                "description": "Relevant tool outputs or evidence for the coordinator.",
            },
            "blockers": {
                "type": "string",
                "description": "Any blockers, uncertainty, or open questions. Use 'none' if none.",
            },
        },
        "required": ["what_did", "key_findings"],
        "additionalProperties": False,
    },
)


def format_executor_report(input_: dict[str, Any]) -> str:
    """Render submit_report input as the same compact report shape used elsewhere."""
    what_did = str(input_.get("what_did", "")).strip()
    key_findings = str(input_.get("key_findings", "")).strip()
    tool_results = str(input_.get("tool_results", "")).strip()
    blockers = str(input_.get("blockers", "")).strip()

    return "\n".join([
        f"- What you did: {what_did or 'not specified'}",
        f"- Key findings / outputs: {key_findings or 'not specified'}",
        f"- Tool results that matter for next steps: {tool_results or 'not specified'}",
        f"- Blockers / open questions: {blockers or 'none'}",
    ])
