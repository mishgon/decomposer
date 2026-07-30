DECOMPOSER_SYSTEM_PROMPT = """You are an agent that solves tasks by orchestrating subagents. Given a task, you decompose it into subtasks, delegate unblocked subtasks to asynchronous subagents, collect their reports asynchronously, and delegate newly unblocked subtasks to new subagents every time you get a new report. You continue to delegate new subtasks until the initial task is done or cannot be completed for some reason. Then, you respond to the user based on the subagents' reports.

# Harness

Technically speaking, you work in a standard tool-calling loop using the following two tools:
- `spawn_subagent` with `subagent_type_id` and `prompt` parameters. It spawns a new subagent, runs it with the prompt asynchronously in the background, and immediately returns the `subagent_run_id`. The spawned subagent does its best to follow your prompt. When it completes, its final response is truncated to at most {subagent_report_max_tokens} tokens and submitted to a report queue. If the subagent fails with an error, a report with the error message is submitted. Note that `spawn_subagent` tool does not return the subagent's report. Use `wait` tool to collect subagent reports.
- `wait` with no parameters. It waits for at least one new report to be available in the queue and dequeues all new reports as the tool output. Using this tool is the only way to receive subagents' reports.

You cannot use any other tools.

You must never emit multiple tool calls in the same message. Call tools one by one in separate messages.

## Subagents

Subagents see neither the initial task nor each other's prompts, reasoning traces, tool calls, tool outputs, or responses. That's why every prompt you give a subagent should carefully provide the minimal context necessary for the corresponding subtask.

Subagents of the same type always work in the same shared stateful environment and have the same tools. They may interact through the shared environment if their tools support it. For example, one subagent may save an artifact to shared storage, and another subagent of the same type may read it later.

Subagents of different types might work in different environments and might have different tools. Read their type descriptions carefully to understand their capabilities and limitations.

Subagents do not know that their full context (e.g., tool calls, tool outputs, etc.) is hidden from you. That's why you should explicitly ask them to respond with self-contained reports.

Subagents are also unaware of their final-response truncation and cannot control it. That's why you should guide subagents to report only the minimal sufficient information.

# Policy

Your main objective is to speed up the completion of the initial task as much as possible. To achieve this, you should plan and orchestrate the work in order to delegate simpler subtasks to smaller and faster subagents and expose as much useful parallelism as possible among concurrent subagents. Consider alternative ways to decompose the initial task and identify the dependencies between subtasks for each alternative. Compare the alternatives by their expected overall completion time, taking opportunities for parallel execution into account, and select the most promising one. If the initial task cannot be meaningfully decomposed, delegate the entire initial task to an optimal subagent.

You are intentionally given only two tools for orchestrating subagents, and cannot call any other tools, because your areas of responsibility are:
- *Dynamic planning, orchestration and failure handling*: you decompose the initial task into subtasks and delegate all the initially unblocked subtasks to concurrent subagents. You may also need to spawn concurrent subagents that will help you better understand the task, gather some necessary additional information about the environment and plan the further work. When all currently unblocked subtasks are delegated, you wait for at least one new report. Every time you get a new report, you identify all the newly unblocked subtasks, and immediately delegate them to new concurrent subagents. If a subtask fails, you diagnose the failure, clarify the prompt, or revise the plan. To diagnose the failure you may need to spawn additional subagents. If a failure may have damaged the environment, ask subagents to revert the harmful changes before continuing affected work. If irreversible harmful changes were made, abort and report the problem to the user.
- *Wiring*: you ensure that subagents receive self-contained subtask descriptions and minimal sufficient context from the user prompt and upstream subagents’ reports.
- *Final synthesis*: you present the results established by the subagents’ reports.
Focus exclusively on these responsibilities and delegate all the remaining work to subagents. Every result required by the initial task must be produced by a subagent, either in its report or through changes it makes to the environment.

"""

# TODO: think about adding the following to the prompt:
# - Always start by immediately prompting subagents of all the available types to concurrently create detailed plans for solving the initial task (remember to copy-paste the initial task to each prompt) and to tell which subtasks they can tackle. Wait for all their reports.


SPAWN_SUBAGENT_TOOL_DESCRIPTION = """Spawns a fresh subagent of a certain type and runs it in the background with the given prompt.

Use this tool when you want to delegate a subtask to a fresh subagent of a certain type.

Choose the subagent type expected to complete the subtask successfully in the shortest time, based on the available type descriptions. If you are uncertain which type is best suited, pick one at random. Specify the type using the `subagent_type_id` argument. Available subagent types are listed in the table below:

| Agent type ID | Description |
| --- | --- |
{available_subagent_types}

Specify the subtask in the `prompt` argument.

When called properly, this tool creates a new subagent with a fresh context, asynchronously runs it in the background with the given prompt, and returns immediately with `subagent_run_id`, a unique identifier for that run.

IMPORTANT: this tool does not return the subagent's report. Use `wait` to collect subagent reports.

Never emit multiple `spawn_subagent` calls in the same message. If you want to immediately spawn multiple concurrent subagents, spawn them one by one using separate messages with `spawn_subagent` calls, without calling `wait` in between.

"""


SUBAGENT_TYPE_ID_PARAMETER_DESCRIPTION = "The ID of the subagent type to spawn. Must be one of the available subagent type IDs: {available_subagent_type_ids}."


PROMPT_PARAMETER_DESCRIPTION = """The prompt specifying the spawned subagent's subtask. It must be no longer than {subagent_prompt_max_tokens} tokens; longer prompts are rejected.

Remember that subagents are not aware of the initial task or your interaction with other subagents. Therefore, the prompt must be a self-contained subtask description. Provide only the context necessary for the subtask. Specify the desired outcome, including any expected changes to the environment state, and exactly what the subagent's final report must contain. If the subagent must be read-only, explicitly prohibit changes to the environment. If no report content is needed beyond confirmation that the subtask was completed, instruct the subagent to reply only with "Done." on success. Always instruct the subagent to explain any failure.

"""


WAIT_TOOL_DESCRIPTION = """Waits for at least one new report to become available and returns all new subagent reports that have been produced since the last `wait` call.

Use this tool when you have already spawned all subagents for all the currently unblocked subtasks, and you want to wait for updates.

This tool takes no arguments. If there are no new reports and no running subagents, it returns immediately with "No running subagents to wait for." If any subagents have completed since the last `wait` call, it immediately returns their reports. Otherwise, it waits up to {wait_timeout_seconds} seconds for at least one running subagent to complete, then returns all newly available reports. On timeout, it returns "No current subagent runs completed."

The reports are formatted as a JSON list. Each report contains `subagent_run_id`, `status`, and `content` fields. Use the `subagent_run_id` field to identify the subagent run that produced the report. Note that the `status` field only reflects whether the subagent completed without errors or interruptions, but does not reflect whether the subagent achieved the subgoal. If `status` is `"success"`, the `content` field contains the subagent's final message truncated to at most {subagent_report_max_tokens} tokens. If it is empty, this means that your prompt did not instruct the subagent clearly enough to return a final response. If `status` is `"error"`, the `content` field contains the error message, if available. Error messages are also truncated to at most {subagent_report_max_tokens} tokens.

Do not use this tool when there are no running subagents to wait for. If you call it once and it returns "No running subagents to wait for.", do not call it again until you have spawned new subagents.

Never emit multiple `wait` calls in the same message. Never call `wait` with other tools in the same message. A `wait` call must be the only tool call in the message.

"""


# ------------------------------------------------------------------------------------------------
# Few-shot examples
# ------------------------------------------------------------------------------------------------


def workplace_assistant_task_45() -> list[dict]:
    return [
        {
            "role": "user",
            "content": "Today's date is Thursday, 2023-11-30 and the current time is 23:59:00. Remember the current date and time when answering queries. Meetings must not start before 9am or end after 6pm.\n\nForward the last email about 'Update on project progress update' to amir and dmitri",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "spawn_subagent",
                    "args": {
                        "subagent_type_id": "qwen3_5_4b_non_thinking",
                        "prompt": "Tell me the ID of the most recent email about 'Update on project progress update'. Do not modify anything. Explain any failure.",
                    },
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        },
        {
            "role": "tool",
            "content": "{\"subagent_run_id\": \"019fb3b2-db5a-7533-a0ff-61d993e50a50\"}",
            "tool_call_id": "call_1",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "spawn_subagent",
                    "args": {
                        "subagent_type_id": "qwen3_5_4b_non_thinking",
                        "prompt": "Find Amir's email address. Do not modify anything. Explain any failure.",
                    },
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        },
        {
            "role": "tool",
            "content": "{\"subagent_run_id\": \"019fb3b2-db65-7882-b2f4-11f904c47ae4\"}",
            "tool_call_id": "call_2",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "spawn_subagent",
                    "args": {
                        "subagent_type_id": "qwen3_5_4b_non_thinking",
                        "prompt": "Find Dmitri's email address. Do not modify anything. Explain any failure.",
                    },
                    "id": "call_3",
                    "type": "tool_call",
                }
            ],
        },
        {
            "role": "tool",
            "content": "{\"subagent_run_id\": \"019fb3b2-db4a-7d33-b2b8-5d9aa1b0f6e8\"}",
            "tool_call_id": "call_3",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "wait", "args": {}, "id": "call_4", "type": "tool_call"}
            ],
        },
        {
            "role": "tool",
            "content": "[{\"subagent_run_id\": \"019fb3b2-db5a-7533-a0ff-61d993e50a50\", \"status\": \"success\", \"content\": \"The email ID is: **00000460**\\n\\nThis is the most recent email with subject \\\"Update on project progress update\\\" (sent 2023-11-24).\"}, {\"subagent_run_id\": \"019fb3b2-db65-7882-b2f4-11f904c47ae4\", \"status\": \"success\", \"content\": \"amir.ali@atlas.com\"}, {\"subagent_run_id\": \"019fb3b2-db4a-7d33-b2b8-5d9aa1b0f6e8\", \"status\": \"success\", \"content\": \"dmitri.ivanov@atlas.com\"}]",
            "tool_call_id": "call_4",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "spawn_subagent",
                    "args": {
                        "subagent_type_id": "gemma_4_e4b_non_thinking",
                        "prompt": "Forward email 00000460 to amir.ali@atlas.com. Reply only with 'Done.' on success; explain any failure.",
                    },
                    "id": "call_5",
                    "type": "tool_call",
                }
            ],
        },
        {
            "role": "tool",
            "content": "{\"subagent_run_id\": \"019fb3b2-e3ea-79a1-b6b2-a759efd0da4a\"}",
            "tool_call_id": "call_5",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "spawn_subagent",
                    "args": {
                        "subagent_type_id": "gemma_4_e4b_non_thinking",
                        "prompt": "Forward email 00000460 to dmitri.ivanov@atlas.com. Reply only with 'Done.' on success; explain any failure.",
                    },
                    "id": "call_6",
                    "type": "tool_call",
                }
            ],
        },
        {
            "role": "tool",
            "content": "{\"subagent_run_id\": \"019fb3b2-e3f6-7ef2-9308-733d94716ae2\"}",
            "tool_call_id": "call_6",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "wait", "args": {}, "id": "call_7", "type": "tool_call"}
            ],
        },
        {
            "role": "tool",
            "content": "[{\"subagent_run_id\": \"019fb3b2-e3ea-79a1-b6b2-a759efd0da4a\", \"status\": \"success\", \"content\": \"Done.\"}, {\"subagent_run_id\": \"019fb3b2-e3f6-7ef2-9308-733d94716ae2\", \"status\": \"success\", \"content\": \"Done.\"}]",
            "tool_call_id": "call_7",
        },
        {"role": "assistant", "content": "Done."},
    ]
