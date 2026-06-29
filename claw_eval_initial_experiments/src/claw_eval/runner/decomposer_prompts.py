"""System prompts for decomposer and executor roles."""

from __future__ import annotations

from typing import Literal

from ..config import PromptConfig
from ..models.task import TaskDefinition
from .decomposer_tools import DECOMPOSER_TOOLS
from .system_prompt import _render_behavior_rules


def _render_executor_tool_guidance(task: TaskDefinition) -> str:
    """Render task-local executor tool guidance for stricter manager ablations."""
    if not task.tools:
        return ""

    lines = [
        "## Executor Environment Tools",
        "The Executor can use only these task tools:",
    ]
    for tool in task.tools:
        description = tool.description.strip() if tool.description else "No description provided."
        lines.append(f"- `{tool.name}`: {description}")

    lines.extend([
        "",
        "When delegating:",
        "- Delegate only subtasks that can be completed with the listed Executor tools.",
        "- Name exact valid tool names in the subtask when the needed tool is obvious.",
        "- Do not ask the Executor to list tools, inspect files or directories, run shell commands, inspect environment variables, or use APIs not listed here.",
        "- After an Executor report-generation failure, retry a smaller subtask using exact valid tool names.",
    ])
    return "\n".join(lines)

def build_decomposer_system_prompt(
    task: TaskDefinition,
    prompt_cfg: PromptConfig | None,
    *,
    manager_valid_tool_guidance: bool = False,
) -> str:
    """Build the decomposer coordinator system prompt."""
    tool_lines = [
        "## Available Control Tools",
        "You cannot call environment tools directly. Use only:",
    ]
    for tool in DECOMPOSER_TOOLS:
        tool_lines.append(f"- {tool.name}: {tool.description}")
    executor_tool_guidance = ""
    if manager_valid_tool_guidance:
        rendered_guidance = _render_executor_tool_guidance(task)
        if rendered_guidance:
            executor_tool_guidance = "\n\n" + rendered_guidance

    behavior = ""
    if prompt_cfg is not None and prompt_cfg.enabled:
        behavior = "\n\n" + _render_behavior_rules(prompt_cfg)

    current_date_lines: list[str] = []
    if task.environment.mock_today:
        current_date_lines = [
            "",
            "## Current Date",
            f"- The benchmark current date is `{task.environment.mock_today}`.",
            "- Use this date for all relative-date reasoning, including today, tomorrow, yesterday, this week, and next week.",
            "- Do not infer dates from model priors or from unrelated examples.",
            "- If a delegated subtask involves a relative date, include the exact absolute date in the subtask.",
        ]

    return "\n".join([
        "You are a Decomposer — a high-level coordinator for a personal assistant benchmark.",
        "",
        "## Role",
        "- You receive the user's original task.",
        "- You break work into natural next steps and delegate executable subtasks to an Executor.",
        "- The Executor can use tools and inspect the environment; you cannot.",
        "- You only observe Executor reports or Executor failure reports, never raw tool outputs.",
        "- After each Executor report, decide whether to delegate another subtask or submit the final answer.",
        *current_date_lines,
        "",
        "## Required Outer Loop",
        "- Your first action must be a native `delegate_subtask` tool call.",
        "- On each later turn, choose exactly one control action: call `delegate_subtask` again or call `submit_final_answer`.",
        "- Each assistant turn must contain exactly one native control tool call, never zero and never more than one.",
        "- Never call both `delegate_subtask` and `submit_final_answer` in the same turn.",
        "- Never call the same control tool twice in one turn.",
        "- Never emit an empty `submit_final_answer`; the answer field must contain the final response for the user.",
        "- Only call `submit_final_answer` after Executor reports contain enough evidence to satisfy the original task.",
        "- If an Executor failure report says no valid report was returned, treat it as a report-generation failure, not evidence that tools or data are unavailable.",
        "- After a report-generation failure, delegate a smaller single-source subtask unless enough evidence is already available.",
        "- Plain text is never a valid control action. Do not describe that you will delegate; actually call `delegate_subtask`.",
        "",
        "## Strategy",
        "- Do not plan the entire solution upfront; react to Executor reports.",
        "- Keep subtasks focused and executable with available tools.",
        "- Include any context the Executor needs directly inside the `subtask` string.",
        "- Prefer one data source per subtask and avoid combining email lookup, finance lookup, and final synthesis in one delegation.",
        "- Do not delegate final report synthesis if Executor reports already contain enough evidence; submit the final answer yourself.",
        "- Do not infer low-level facts from missing reports; ask the Executor for the next needed evidence.",
        "- Preserve safety constraints and verification steps from the original task.",
        "- Your final answer must fully satisfy the user's original request.",
        "",
        "## Delegation Report Contract",
        "- Delegate natural-language subtasks, not raw environment-tool commands.",
        "- Ask the Executor to return concrete completion evidence in its report: relevant IDs, dates, names, created or updated artifact IDs, exact values, and blockers.",
        "- Do not ask for raw tool outputs, raw JSON, logs, traces, hidden prompts, or tool internals.",
        "- Do not require the Executor to expose implementation details; verify progress from its summarized report artifacts.",
        "- Prefer subtasks whose expected report can be checked against the original user requirements.",
        "",
        "## Final Answer Checklist",
        "Before calling `submit_final_answer`, verify all of the following:",
        "- Every numbered or explicit requirement in the original user task is satisfied.",
        "- Required side effects were completed, not merely planned.",
        "- Executor reports contain concrete evidence for each requirement, such as IDs, dates, names, saved draft IDs, updated records, exact values, or audit-relevant artifacts.",
        "- No requirement depends only on an Executor failure report or missing report.",
        "- If any required evidence or side effect is missing, call `delegate_subtask` for one smaller subtask instead of submitting a final answer.",
        "",
        tool_lines[0],
        *tool_lines[1:],
        "",
        "Tool-call protocol is strict: use native API tool/function calls only.",
        "Never simulate tool calls as plain text markup.",
        executor_tool_guidance,
        behavior,
    ]).strip()


def build_executor_subtask_user_content(
    subtask: str,
    context: str = "",
    *,
    report_tool_name: str | None = None,
    prompt_mode: Literal["report_wrapper", "flat_subtask"] = "report_wrapper",
) -> str:
    """Build the initial user message for an executor subtask."""
    _ = context  # Kept for compatibility with old traces/tests; new prompts use one subtask string.
    if prompt_mode == "flat_subtask":
        return subtask.strip()
    if prompt_mode != "report_wrapper":
        raise ValueError(f"Unknown executor prompt mode: {prompt_mode!r}")

    parts = [
        "You are the Executor. Complete exactly this subtask using the available tools.",
        "",
        "Subtask:",
        subtask.strip(),
        "",
        "First use the available tools to complete the subtask.",
    ]
    if report_tool_name:
        parts.extend([
            f"When finished, call the native `{report_tool_name}` tool exactly once with a compact report.",
            "Call it alone after observing relevant tool results; plain text final reports are invalid in this mode.",
            "Populate the report fields with:",
        ])
    else:
        parts.append("When finished, reply with a compact report containing:")
    parts.extend([
        "- What you did",
        "- Key findings / outputs",
        "- Any blockers or open questions",
    ])
    return "\n".join(parts)
