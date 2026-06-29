"""Synthetic control tools for the decomposer outer loop."""

from __future__ import annotations

from ..models.tool import ToolSpec

DELEGATE_SUBTASK = ToolSpec(
    name="delegate_subtask",
    description=(
        "Delegate an executable subtask to the Executor agent. "
        "The Executor can use environment tools and returns a compact report or an explicit failure report. "
        "Write the subtask as clear, self-contained instructions, including any context the Executor needs."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "subtask": {
                "type": "string",
                "description": "Natural-language subtask for the Executor to complete.",
            },
        },
        "required": ["subtask"],
        "additionalProperties": False,
    },
)

SUBMIT_FINAL_ANSWER = ToolSpec(
    name="submit_final_answer",
    description=(
        "Submit the final user-facing answer when the original task is complete. "
        "Include citations, safety notes, and verification when relevant."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Final response to the end user.",
            },
        },
        "required": ["answer"],
    },
)

DECOMPOSER_TOOLS: list[ToolSpec] = [DELEGATE_SUBTASK, SUBMIT_FINAL_ANSWER]

DELEGATE_SUBTASK_NAME = DELEGATE_SUBTASK.name
SUBMIT_FINAL_ANSWER_NAME = SUBMIT_FINAL_ANSWER.name
