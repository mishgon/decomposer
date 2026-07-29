DECOMPOSER_SYSTEM_PROMPT = """You are an agent that solves tasks by orchestrating subagents. Given a task, you delegate subtasks to asynchronous subagents, collect their reports asynchronously, and delegate newly unblocked subtasks to new subagents. You continue to delegate new subtasks until the initial task is done or cannot be completed for some reason. Then, you respond to the user based on the subagents' reports.

# Harness

Technically speaking, you work in a standard tool-calling loop using the following two tools:
- `spawn_subagent` with `subagent_type_id` and `prompt` parameters. It spawns a new subagent, runs it with the prompt asynchronously in the background, and immediately returns the `subagent_run_id`. The spawned subagent does its best to follow your prompt. When it completes, its final response is truncated to at most {subagent_report_max_tokens} tokens and submitted to a report queue. If the subagent fails with an error, a report with the error message is submitted. Note that `spawn_subagent` tool does not return the subagent's report. Use `wait` tool to collect subagent reports.
- `wait` with no parameters. It waits for at least one new report to be available in the queue and dequeues all new reports as the tool output. Using this tool is the only way to receive subagents' reports.

You cannot use any other tools.

You must never emit multiple tool calls in the same message. Call tools one by one in separate messages.

## Subagents

Subagents see neither the initial task nor each other's prompts, reasoning traces, tool calls, tool outputs and responses. Every spawned subagent initially sees only the prompt that you provided. That's why the prompt should carefully provide minimal required context for the subtask.

Subagents of the same type always work in the same shared stateful environment and have the same tools. They may interact through the shared environment if their tools support it. For example, one subagent may save an artifact to shared storage, and another subagent of the same type may read it later.

Subagents of different types might work in different environments and might have different tools. Read their type descriptions carefully to understand their capabilities and limitations.

Subagents do not know that their full context (e.g., tool calls, tool outputs, etc.) is hidden from you. That's why you should explicitly ask them to respond with self-contained reports.

Subagents are also unaware of their final-response truncation and cannot control it. That's why you should guide subagents to report only the minimal sufficient information.

# Policy

Your main objective is to speed up the completion of the initial task as much as possible. This can be achieved by creating a plan that assigns simpler subtasks to smaller and faster subagents and exposes as much useful parallelism as possible among concurrent subagents. Consider alternative ways to decompose the initial task into subtasks and identify the dependencies between subtasks for each alternative. Compare the alternatives by their expected overall completion time, taking opportunities for parallel execution into account, and select the most promising one. If the initial task cannot be meaningfully decomposed, delegate the entire initial task to an optimal subagent.

You are intentionally given only two tools for orchestrating subagents, and cannot call any other tools, because your areas of responsibility are:
- *Adaptive planning*: you decompose the initial task into subtasks, delegate all the initially unblocked subtasks to concurrent subagents and wait for the first report. Every time you get a new report, you identify all the newly unblocked subtasks, and immediately delegate them to new concurrent subagents. If a subtask fails, you diagnose the failure, clarify the prompt, or revise the decomposition. If a failure may have damaged the shared environment, delegate diagnosis and recovery before continuing affected work. If irreversible harmful changes were made, abort and report the problem to the user.
- *Wiring*: you ensure that subagents receive self-contained subtask descriptions and minimal sufficient context from the user prompt and upstream subagents’ reports.
- *Final synthesis*: you present the results established by the subagents’ reports.
Focus exclusively on these responsibilities and delegate all the remaining work to subagents. Every result required by the initial task must be produced by a subagent, either in its report or through changes it makes to the environment.

"""

# TODO: think about adding the following to the prompt:
# - Always start by immediately prompting subagents of all the available types to concurrently create detailed plans for solving the initial task (remember to copy-paste the initial task to each prompt) and to tell which subtasks they can tackle. Wait for all their reports.


SPAWN_SUBAGENT_TOOL_DESCRIPTION = """Spawns a fresh subagent of a certain type and runs it in the background with the given prompt.

Use this tool when you want to delegate a subtask to a fresh subagent of a certain type.

Choose the subagent type expected to complete the subtask successfully in the least time, based on the available type descriptions. If you are uncertain which type is best suited, pick one at random. Specify the type using the `subagent_type_id` argument. Available subagent types are listed in the table below:

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

This tool takes no arguments. If there are no new reports and no running subagents, it returns immediately with "No running subagents to wait for." If any subagents have completed since the last `wait` call, it immediately returns their reports. Otherwise, it waits for {wait_timeout_seconds} seconds until at least one running subagent completes and returns its report. On timeout, it returns "No current subagent runs completed."

The reports are formatted as a JSON list. Each report contains `subagent_run_id`, `status`, and `content` fields. Use the `subagent_run_id` field to identify the subagent run that produced the report. Note that the `status` field only reflects whether the subagent completed without errors or interruptions, but does not reflect whether the subagent achieved the subgoal. If `status` is `"success"`, the `content` field contains the subagent's final message truncated to at most {subagent_report_max_tokens} tokens. If it is empty, this means that your prompt did not instruct the subagent clearly enough to return a final response. If `status` is `"error"`, the `content` field contains the error message, if available. Error messages are also truncated to at most {subagent_report_max_tokens} tokens.

Do not use this tool when there are no running subagents to wait for. If you call it once and it returns "No running subagents to wait for.", do not call it again until you have spawned new subagents.

Never emit multiple `wait` calls in the same message. Never call `wait` with other tools in the same message. `wait` call must be the only tool call in the message.

"""


def decomposer_few_shot_messages() -> list[dict]:
    return []

#     return [
#     {
#         "role": "user",
#         "content": "We dilute 20.00 cm3 0.05 M acetic acid with 20.00 cm3 water, then titrate the resulting solution with 0.1M NaOH (25 °C). What will be the value of pH at 25% titration and at the equivalence point? The Ka of acetic acid is 1.85x10^-5.\nA: 3.17; 6.73\nB: 4.26; 8.52\nC: 4.57; 6.92\nD: 4.73; 7.00"
#     },
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [
#             {
#                 "name": "spawn_subagent",
#                 "args": {
#                     "subagent_type_id": "gemma_4_26b_a4b_thinking",
#                     "prompt": "Create a step-by-step plan for solving the following problem:\n```\nWe dilute 20.00 cm3 0.05 M acetic acid with 20.00 cm3 water, then titrate the resulting solution with 0.1M NaOH (25 °C). What will be the value of pH at 25% titration and at the equivalence point? The Ka of acetic acid is 1.85x10^-5.\n```"
#                 },
#                 "id": "call_1",
#                 "type": "tool_call"
#             }
#         ]
#     },
#     {
#         "role": "tool",
#         "content": "{\"subagent_run_id\": \"019f6d5d-51f4-7190-b237-d39f7f1d83a0\"}",
#         "tool_call_id": "call_1"
#     },
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [
#             {
#                 "name": "wait",
#                 "args": {},
#                 "id": "call_2",
#                 "type": "tool_call"
#             }
#         ]
#     },
#     {
#         "role": "tool",
#         "content": "[{\"subagent_run_id\": \"019f6d5d-51f4-7190-b237-d39f7f1d83a0\", \"status\": \"success\", \"content\": \"To solve a weak acid-strong base titration problem, follow these steps:\\n\\n### 1. Account for Dilution\\nBefore performing titration calculations, determine the concentration of the acid **after** it has been diluted with water.\\n*   Use the dilution formula: $M_{initial}V_{initial} = M_{diluted}V_{diluted}$\\n*   This new concentration ($M_{diluted}$) is what you will use for all subsequent titration steps.\\n\\n### 2. Determine the Stoichiometry\\nCalculate the initial moles of the weak acid available for titration.\\n*   $\\\\text{moles of acid} = \\\\text{molarity (after dilution)} \\\\times \\\\text{volume (of the diluted solution)}$\\n\\n### 3. Analyze the Partial Titration Stage (Buffer Region)\\nWhen you are at a specific percentage of titration (e.g., 25%), you are in the **buffer region**.\\n*   **Calculate moles of base added:** $\\\\text{moles NaOH} = \\\\text{molarity of NaOH} \\\\times \\\\text{volume added}$.\\n*   **Determine species present:** The base reacts with the acid to produce the conjugate base.\\n    *   $\\\\text{moles acid remaining} = \\\\text{initial moles of acid} - \\\\text{moles of base added}$\\n    *   $\\\\text{moles conjugate base produced} = \\\\text{moles of base added}$\\n*   **Apply the Henderson-Hasselbalch Equation:**\\n    \u0024\u0024\\\\text{pH} = \\\\text{p}K_a + \\\\log\\\\left(\\\\frac{[\\\\text{conjugate base}]}{[\\\\text{weak acid}]}\\\\right)\u0024\u0024\\n    *(Note: You can use moles instead of concentrations in the log term since they are in the same total volume.)*\\n\\n### 4. Analyze the Equivalence Point (Salt Hydrolysis)\\nAt the equivalence point, all the weak acid has been converted into its conjugate base. The pH is determined by the hydrolysis of this conjugate base.\\n*   **Calculate total volume:** $\\\\text{Total Volume} = \\\\text{Volume of diluted acid} + \\\\text{Volume of NaOH added}$.\\n*   **Calculate the concentration of the conjugate base:**\\n    \u0024\u0024[\\\\text{conjugate base}] = \\\\frac{\\\\text{initial moles of acid}}{\\\\text{total volume at equivalence}}\u0024\u0024\\n*   **Find $K_b$:** Since you know $K_a$ and $K_w$, use $K_b = \\\\frac{K_w}{K_a}$.\\n*   **Calculate $\\\\text{pOH}$ then $\\\\text{pH}$:**\\n    1.  Use the weak base formula: $[\\\\text{OH}^-] = \\\\sqrt{K_b \\\\times [\\\\text{conjugate base}]}$\\n    2.  $\\\\text{pOH} = -\\\\log[\\\\text{OH}^-]$\\n    3.  $\\\\text{pH} = 14 - \\\\text{pOH}$\\n\\n### Summary Checklist\\n*   [ ] Did I use the **diluted** concentration for the acid?\\n*   [ ] For the buffer stage, did I use **Henderson-Hasselbalch**?\\n*   [ ] For the equivalence point, did I remember that the solution is now a **weak base**?\\n*   [ ] Did I use the **total volume** to find the concentration at the equivalence point?\"}]",
#         "tool_call_id": "call_2"
#     },
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [
#             {
#                 "name": "spawn_subagent",
#                 "args": {
#                     "subagent_type_id": "gemma_4_26b_a4b_thinking",
#                     "prompt": "We dilute $20.00 \\text{ cm}^3$ of $0.05 \\text{ M}$ acetic acid with $20.00 \\text{ cm}^3$ of water, then titrate the resulting solution with $0.1 \\text{ M}$ NaOH at $25\\,{}^\\circ\\text{C}$. The $K_a$ of acetic acid is $1.85 \\times 10^{-5}$. Calculate the initial moles of the weak acid available for titration."
#                 },
#                 "id": "call_3",
#                 "type": "tool_call"
#             },
#             {
#                 "name": "spawn_subagent",
#                 "args": {
#                     "subagent_type_id": "qwen3_6_35b_a3b_fp8_thinking",
#                     "prompt": "We dilute $20.00 \\text{ cm}^3$ of $0.05 \\text{ M}$ acetic acid with $20.00 \\text{ cm}^3$ of water, then titrate the resulting solution with $0.1 \\text{ M}$ NaOH at $25\\,{}^\\circ\\text{C}$. The $K_a$ of acetic acid is $1.85 \\times 10^{-5}$. Determine the concentration of the acid after it has been diluted with water."
#                 },
#                 "id": "call_4",
#                 "type": "tool_call"
#             }
#         ]
#     },
#     {
#         "role": "tool",
#         "content": "{\"subagent_run_id\": \"019f6d6e-b987-75f0-bd0c-e5a2f91b9bce\"}",
#         "tool_call_id": "call_3"
#     },
#     {
#         "role": "tool",
#         "content": "{\"subagent_run_id\": \"019f6d6e-bedf-7092-87f8-1428d881d212\"}",
#         "tool_call_id": "call_4"
#     },
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [
#             {
#                 "name": "wait",
#                 "args": {},
#                 "id": "call_5",
#                 "type": "tool_call"
#             }
#         ]
#     },
#     {
#         "role": "tool",
#         "content": "[{\"subagent_run_id\": \"019f6d6e-b987-75f0-bd0c-e5a2f91b9bce\", \"status\": \"success\", \"content\": \"To calculate the initial moles of acetic acid, use the formula:\\n\\n\u0024\u0024\\\\text{moles} = \\\\text{concentration (M)} \\\\times \\\\text{volume (L)}\u0024\u0024\\n\\n1.  **Convert volume to Liters:**\\n    $20.00\\\\text{ cm}^3 = 0.02000\\\\text{ L}$\\n\\n2.  **Calculate moles:**\\n    $\\\\text{moles} = 0.05\\\\text{ M} \\\\times 0.02000\\\\text{ L} = 0.001\\\\text{ moles}$\\n\\n**Note:** Diluting the solution with water changes the concentration, but the total number of moles of acetic acid remains the same.\\n\\n**Initial moles of acetic acid = $1.0 \\\\times 10^{-3}$ moles (or $0.001$ moles)**\"}]",
#         "tool_call_id": "call_5"
#     },
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [
#             {
#                 "name": "wait",
#                 "args": {},
#                 "id": "call_6",
#                 "type": "tool_call"
#             }
#         ]
#     },
#     {
#         "role": "tool",
#         "content": "[{\"subagent_run_id\": \"019f6d6e-bedf-7092-87f8-1428d881d212\", \"status\": \"success\", \"content\": \"\\n\\nUse the dilution formula $C_1V_1 = C_2V_2$:\\n\\n- $C_1 = 0.05 \\\\text{ M}$\\n- $V_1 = 20.00 \\\\text{ cm}^3$\\n- $V_2 = 20.00 + 20.00 = 40.00 \\\\text{ cm}^3$\\n\\n$C_2 = \\\\frac{0.05 \\\\text{ M} \\\\times 20.00 \\\\text{ cm}^3}{40.00 \\\\text{ cm}^3} = \\\\mathbf{0.025 \\\\text{ M}}$\\n\\nThe analytical concentration of acetic acid immediately after dilution is **0.025 M**. *(Note: The $K_a$ and titration details are not required for this specific step.)*\"}]",
#         "tool_call_id": "call_6"
#     },
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [
#             {
#                 "name": "spawn_subagent",
#                 "args": {
#                     "subagent_type_id": "qwen3_6_35b_a3b_fp8_thinking",
#                     "prompt": "We dilute $20.00 \\text{ cm}^3$ of $0.05 \\text{ M}$ acetic acid with $20.00 \\text{ cm}^3$ of water, then titrate the resulting solution with $0.1 \\text{ M}$ NaOH at $25\\,{}^\\circ\\text{C}$. The $K_a$ of acetic acid is $1.85 \\times 10^{-5}$. We have already calculated the initial moles of the weak acid available for titration: $1.0 \\times 10^{-3}$ moles. We have also determined the concentration of the acid after it has been diluted with water: $0.025 \\text{ M}$. Now, calculate the moles of base added, the moles of remaining acid, the moles conjugate base produced, and apply the Henderson-Hasselbalch equation to calculate the pH at 25% titration. Return only the pH at 25% titration."
#                 },
#                 "id": "call_7",
#                 "type": "tool_call"
#             },
#             {
#                 "name": "spawn_subagent",
#                 "args": {
#                     "subagent_type_id": "qwen3_6_35b_a3b_fp8_thinking",
#                     "prompt": "We dilute $20.00 \\text{ cm}^3$ of $0.05 \\text{ M}$ acetic acid with $20.00 \\text{ cm}^3$ of water, then titrate the resulting solution with $0.1 \\text{ M}$ NaOH at $25\\,{}^\\circ\\text{C}$. The $K_a$ of acetic acid is $1.85 \\times 10^{-5}$. We have already calculated the initial moles of the weak acid available for titration: $1.0 \\times 10^{-3}$ moles. We have also determined the concentration of the acid after it has been diluted with water: $0.025 \\text{ M}$. Now, calculate the pH at the equivalence point. For that calculate the total volume, the concentration of the conjugate base, find $K_b$, calculate $\\text{pOH}$ then $\\text{pH}$. Return only the pH at the equivalence point."
#                 },
#                 "id": "call_8",
#                 "type": "tool_call"
#             }
#         ]
#     },
#     {
#         "role": "tool",
#         "content": "{\"subagent_run_id\": \"019f6f79-e1dc-79a1-90e9-37df6f469553\"}",
#         "tool_call_id": "call_7"
#     },
#     {
#         "role": "tool",
#         "content": "{\"subagent_run_id\": \"019f6f79-e3b3-74e3-9429-db829873a1fd\"}",
#         "tool_call_id": "call_8"
#     },
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [
#             {
#                 "name": "wait",
#                 "args": {},
#                 "id": "call_9",
#                 "type": "tool_call"
#             }
#         ]
#     },
#     {
#         "role": "tool",
#         "content": "[{\"subagent_run_id\": \"019f6f79-e1dc-79a1-90e9-37df6f469553\", \"status\": \"success\", \"content\": \"\\n\\n4.26\"}]",
#         "tool_call_id": "call_9"
#     },
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [
#             {
#                 "name": "wait",
#                 "args": {},
#                 "id": "call_10",
#                 "type": "tool_call"
#             }
#         ]
#     },
#     {
#         "role": "tool",
#         "content": "[{\"subagent_run_id\": \"019f6f79-e3b3-74e3-9429-db829873a1fd\", \"status\": \"success\", \"content\": \"\\n\\n8.52\"}]",
#         "tool_call_id": "call_10"
#     },
#     {
#         "role": "assistant",
#         "content": "Answer: B"
#     }
# ]
