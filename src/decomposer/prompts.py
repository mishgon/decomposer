from importlib.resources import files


DECOMPOSER_TEACHER_SYSTEM_PROMPT = (
    files("decomposer")
    .joinpath("teacher_system_prompt.md")
    .read_text(encoding="utf-8")
    .strip()
)


DECOMPOSER_SYSTEM_PROMPT = """You are a manager agent. Complete user tasks exclusively by orchestrating subagents through the provided tools."""


SPAWN_SUBAGENT_TOOL_DESCRIPTION = """Creates a new subagent of a certain type with a fresh context, asynchronously runs it in the background with the given prompt, and returns immediately with a unique identifier for that run.

Use this tool when you want to delegate a subtask to a fresh subagent of a certain type.

Specify the subagent type using the `subagent_type_id` parameter and the subtask using the `prompt` parameter. Note that this tool returns only the run ID and does not return the subagent report. Use the `wait` tool to collect subagent reports."""


SUBAGENT_TYPE_ID_PARAMETER_DESCRIPTION = """The ID of the subagent type to spawn.

Available subagent types are listed in the table below:
| Agent type ID | Description |
| --- | --- |
{available_subagent_types}"""


PROMPT_PARAMETER_DESCRIPTION = """The prompt specifying the subtask. It must be at most {subagent_prompt_max_tokens} approximate tokens."""


WAIT_TOOL_DESCRIPTION = """Waits for at least one new report to become available and returns all new subagent reports that have been produced since the last `wait` call.

Use this tool when you have already spawned subagents for all the necessary unblocked subtasks, and you want to wait for updates.

This tool takes no arguments. If there are no new reports and no running subagents, it returns immediately with "{no_running_subagents_error}" If any subagents have completed since the last `wait` call, it immediately returns their reports. Otherwise, it waits up to {wait_timeout_seconds} seconds for at least one running subagent to complete, then returns all newly available reports. On timeout, it returns "{no_completed_subagent_runs_error}"

The reports are formatted as a JSON list. Each report contains `subagent_run_id`, `status`, and `content` fields. Use the `subagent_run_id` field to identify the subagent run that produced the report. Note that the `status` field only reflects the mechanical status of the run, but does not reflect whether the subagent completed the subtask successfully. If status is `"success"`, the `content` field contains the subagent's final response, truncated to at most {subagent_report_max_tokens} approximate tokens. If status is `"error"`, the `content` field contains the error message, if available, with the same limit.

Do not use the `wait` tool when there are no running subagents to wait for.

Never emit multiple `wait` calls in the same message. Never call `wait` with other tools in the same message. A `wait` call must be the only tool call in the message."""


NO_RUNNING_SUBAGENTS_ERROR = "No running subagents to wait for."


NO_COMPLETED_SUBAGENT_RUNS_ERROR = "No current subagent runs completed."


PARALLEL_WAIT_CALL_ERROR = """A `wait` call must be the only tool call in the message. This `wait` call was not executed."""


EARLY_REPORT_ERROR = """Wait for the reports of all spawned subagents. If everything is OK, respond with a final report again; otherwise, continue the orchestration until reports of *all* spawned subagents collectively establish completion of the user task."""


EMPTY_REPORT_ERROR = """Respond again with a non-empty final report."""


SUBAGENT_SYSTEM_PROMPT = """You are a helpful assistant. Complete user tasks using provided tools. Always respond to the user with a final report upon task completion. If you fail to fully complete a task, report what was accomplished and state the failure reason. Keep your reports as concise as possible (but always not empty) while satisfying these guardrails and user prompt constraints."""
