import asyncio
import json
import logging
import time
from typing import Annotated, Any, NotRequired, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT
from langchain.tools import ToolRuntime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Command
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


SUBAGENT_PROMPT_MAX_TOKENS = 1024
SUBAGENT_REPORT_MAX_TOKENS = 1024
DECOMPOSER_SYSTEM_PROMPT = f"""You are a Decomposer agent.

You initially see only the user prompt specifying your goal in the environment. You cannot touch the environment directly; you interact with it exclusively via spawning and prompting subagents and reading their reports. Decompose your goal into immediate subgoals or tasks, prompt subagents to achieve / execute them, read their reports, define next subgoals / tasks, prompt new subagents, etc., until you are sure that your main goal has been achieved.

Specifically, you can interact with the environment by calling the following tools:
- `spawn_subagent` with `subagent_type_id` and `prompt` args. It spawns a fresh subagent and runs it with the given prompt asynchronously in the background. You can imagine that the spawned subagents have their own isolated contexts and tools for reasoning and interacting with the environment, do their best to follow your prompts, and finally write and submit their reports to a shared queue. If a subagent fails with an error, it also submits a report with the error message.
- `wait` with no args. It waits for at least one subagent report to be available in the queue and dequeues all reports from the queue as the tool output.

Note that each report contains only the subagent's final message, while the remaining subagent's context, i.e. prompt, tool calls, and tool outputs, are not included. Moreover, the subagent's final message is always truncated to at most {SUBAGENT_REPORT_MAX_TOKENS} tokens. Subagents are not responsible for this truncation and cannot control it. This is intentional to prevent your context rot and overflow. Still, you can ask subagents to include any information within the token limits in their final responses.

Follow these general principles:
- Delegate all the work that requires domain-specific knowledge or expertise to the appropriate subagents. Your role is only to optimally decompose the goals / tasks into subgoals / subtasks and manage the subagents to achieve / execute them. Still, you are responsible for the final result.
- Decompose goals / tasks to independent subgoals / subtasks that can be achieved / executed *in parallel* and assign them to concurrent subagents whenever possible. This can significantly speed up the work.
- Stop spawning and waiting for subagents when the main goal has been achieved and respond to the user.
"""


SPAWN_SUBAGENT_TOOL_DESCRIPTION_TEMPLATE = """Spawns a fresh subagent of a certain type and runs it in the background with the given prompt.

Use this tool when you want to set a subgoal or delegate a task (e.g., gather a piece of information or change the environmental state in a specific way) to a fresh subagent of a certain type.

Depending on the subgoal / task, select the subagent type best suited to handle it. First optimize quality, then cost. Specify the type using the `subagent_type_id` argument. Available subagent types are listed in the table below:

| Agent type ID | Description |
| --- | --- |
{available_subagent_types}

Specify the subgoal / task in the `prompt` argument. It can be any free-form text (e.g., goal, task, question, query, instructions, etc.) up to {subagent_prompt_max_tokens} tokens (longer prompts are rejected). A good prompt states the subagent's goal / task, provides minimal required context and explicitly describes what information the subagent's final response must or must not include. Note that subagents might not know that their context is isolated from yours and that you receive only their final response truncated to at most {subagent_report_max_tokens} tokens. It is your responsibility to instruct subagents to produce an informative final response no longer than {subagent_report_max_tokens} tokens.

When called properly, this tool creates a new subagent with a fresh context, asynchronously runs it in the background, and returns immediately with `subagent_run_id`, a unique identifier for the subagent run.

IMPORTANT: this tool does not return the subagent's report. Use `wait` to collect subagent reports.

You can call this tool multiple times (in a single message or separate messages) to asynchronously spawn multiple concurrent subagents without waiting for the previously spawned subagents.

Spawn as many subagents as needed to achieve the main goal, but no more than necessary.
"""

WAIT_TIMEOUT_SECONDS = 60.0
WAIT_TOOL_DESCRIPTION = f"""Waits for at least one new report to become available and returns all new subagent reports that have been produced since the last `wait` call.

Use this tool when you have already spawned all subagents you find necessary at the moment, and you want to wait for updates.

This tool takes no arguments. If there are no new reports and no running subagents, it returns immediately with "No running subagents to wait for." If any subagents have completed since the last `wait` call, it immediately returns their reports. Otherwise, it waits for {WAIT_TIMEOUT_SECONDS} seconds until at least one running subagent completes and returns its report. On timeout, it returns "No current subagent runs completed."

The reports are formatted as a JSON list. Each report contains `subagent_run_id`, `status`, and `content` fields. Use the `subagent_run_id` field to identify the subagent run that produced the report. Note that the `status` field only reflects whether the subagent has completed without errors or interruptions, but does not reflect whether the subagent achieved the subgoal. If `status` is `"success"`, the `content` field contains the subagent's final message truncated to at most {SUBAGENT_REPORT_MAX_TOKENS} tokens. If it is empty, this means that your prompt did not instruct the subagent clearly enough to return a final response. If `status` is `"error"`, the `content` field contains the error message, if any. Error messages are also truncated to at most {SUBAGENT_REPORT_MAX_TOKENS} tokens.

Do not use this tool when there are no running subagents to wait for. If you call it once and it returns "No running subagents to wait for.", do not call it again until you have spawned new subagents.
"""


SYNC_WAIT_POLL_SECONDS = 5.0
TERMINAL_STATUSES = frozenset({"success", "error", "timeout", "interrupted"})
SUBAGENT_RECURSION_LIMIT = 200  # ~100 tool calls
HISTORY_LIMIT = 1000


class SubagentType(TypedDict):
    subagent_type_id: str
    description: str
    assistant_id: str
    url: NotRequired[str]
    headers: NotRequired[dict[str, str]]


class SubagentReport(TypedDict):
    subagent_run_id: str
    status: str
    content: str | None


class SubagentRun(TypedDict):
    subagent_run_id: str
    # TODO: Add `subagent_id` when `invoke_subagent` is introduced. It should
    # identify a persistent subagent thread; `subagent_run_id` identifies one run.
    subagent_type_id: str
    assistant_id: str
    thread_id: str
    run_id: str
    status: str
    tool_call_count: NotRequired[int | None]
    prompt: str
    report: NotRequired[SubagentReport | None]


def _subagent_runs_reducer(
    existing: dict[str, SubagentRun] | None,
    update: dict[str, SubagentRun],
) -> dict[str, SubagentRun]:
    merged = dict(existing or {})
    merged.update(update)
    return merged


def _build_spawn_subagent_schema(subagent_types: dict[str, SubagentType]) -> type[BaseModel]:
    available_subagent_type_ids = ", ".join(f"`{k}`" for k in subagent_types)

    class SpawnSubagentSchema(BaseModel):
        subagent_type_id: str = Field(
            description=(
                "The ID of the subagent type to spawn. Must be one of the "
                f"available subagent type IDs: {available_subagent_type_ids}."
            )
        )
        prompt: str = Field(
            description=(
                "The subgoal / task prompt to send to the spawned subagent. "
                f"Must be no longer than {SUBAGENT_PROMPT_MAX_TOKENS} tokens. "
                "Specifies what the subagent should do and what information "
                "its final response must or must not include."
            )
        )

    return SpawnSubagentSchema


class DecomposerAgentState(AgentState[ResponseT]):
    subagent_runs: Annotated[NotRequired[dict[str, SubagentRun]], _subagent_runs_reducer]


def _count_text_tokens(text: str) -> int:
    return count_tokens_approximately([{"role": "user", "content": text}])


def _truncate_text(text: str | None, max_tokens: int) -> tuple[str | None, bool]:
    if text is None or _count_text_tokens(text) <= max_tokens:
        return text, False

    suffix = f"\n\n[truncated to approximately {max_tokens} tokens]"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + suffix
        if _count_text_tokens(candidate) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + suffix, True


def _resolve_headers(subagent_type: SubagentType) -> dict[str, str]:
    headers: dict[str, str] = dict(subagent_type.get("headers") or {})
    if "x-auth-scheme" not in headers:
        headers["x-auth-scheme"] = "langsmith"
    return headers


class _ClientCache:
    """Adapted from deepagents.middleware.async_subagents.ClientCache."""

    def __init__(self, subagent_types: dict[str, SubagentType]) -> None:
        self._subagent_types = subagent_types
        self._sync: dict[tuple[str | None, frozenset[tuple[str, str]]], SyncLangGraphClient] = {}
        self._async: dict[tuple[str | None, frozenset[tuple[str, str]]], LangGraphClient] = {}

    def _cache_key(self, subagent_type: SubagentType) -> tuple[str | None, frozenset[tuple[str, str]]]:
        return (subagent_type.get("url"), frozenset(_resolve_headers(subagent_type).items()))

    def get_sync(self, subagent_type_id: str) -> SyncLangGraphClient:
        subagent_type = self._subagent_types[subagent_type_id]
        if subagent_type.get("url") is None:
            msg = f"Subagent type '{subagent_type_id}' has no url configured. ASGI transport (url=None) requires async invocation."
            raise ValueError(msg)
        key = self._cache_key(subagent_type)
        if key not in self._sync:
            self._sync[key] = get_sync_client(
                url=subagent_type.get("url"),
                headers=_resolve_headers(subagent_type),
            )
        return self._sync[key]

    def get_async(self, subagent_type_id: str) -> LangGraphClient:
        subagent_type = self._subagent_types[subagent_type_id]
        key = self._cache_key(subagent_type)
        if key not in self._async:
            self._async[key] = get_client(
                url=subagent_type.get("url"),
                headers=_resolve_headers(subagent_type),
            )
        return self._async[key]


def _build_spawn_subagent_tool_description(
    subagent_types: dict[str, SubagentType],
) -> str:
    subagent_types_desc = "\n".join(
        f"| `{subagent_type_id}` | {subagent_type['description']} |"
        for subagent_type_id, subagent_type in subagent_types.items()
    )
    return SPAWN_SUBAGENT_TOOL_DESCRIPTION_TEMPLATE.format(
        subagent_prompt_max_tokens=SUBAGENT_PROMPT_MAX_TOKENS,
        subagent_report_max_tokens=SUBAGENT_REPORT_MAX_TOKENS,
        available_subagent_types=subagent_types_desc
    )


def _build_spawn_subagent_tool(
    subagent_types: dict[str, SubagentType],
    clients: _ClientCache,
) -> StructuredTool:

    def spawn_subagent(
        subagent_type_id: str,
        prompt: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type_id not in subagent_types:
            allowed = ", ".join(f"`{k}`" for k in subagent_types)
            return f"Unknown subagent type ID `{subagent_type_id}`. Available IDs: {allowed}."

        prompt_token_count = _count_text_tokens(prompt)
        if prompt_token_count > SUBAGENT_PROMPT_MAX_TOKENS:
            return f"The prompt is too long (about {prompt_token_count} tokens) while the limit is {SUBAGENT_PROMPT_MAX_TOKENS} tokens."

        subagent_type = subagent_types[subagent_type_id]
        client = clients.get_sync(subagent_type_id)
        thread = client.threads.create()
        run = client.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=subagent_type["assistant_id"],
            input={"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": SUBAGENT_RECURSION_LIMIT},
        )
        subagent_run_id = run["run_id"]
        subagent_run: SubagentRun = {
            "subagent_run_id": subagent_run_id,
            "subagent_type_id": subagent_type_id,
            "assistant_id": subagent_type["assistant_id"],
            "thread_id": thread["thread_id"],
            "run_id": run["run_id"],
            "status": run["status"],
            "prompt": prompt,
        }
        tool_output: dict[str, Any] = {
            "subagent_run_id": subagent_run_id,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(tool_output, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "subagent_runs": {subagent_run_id: subagent_run},
            }
        )

    async def aspawn_subagent(
        subagent_type_id: str,
        prompt: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type_id not in subagent_types:
            allowed = ", ".join(f"`{k}`" for k in subagent_types)
            return f"Unknown subagent type ID `{subagent_type_id}`. Available IDs: {allowed}."

        prompt_token_count = _count_text_tokens(prompt)
        if prompt_token_count > SUBAGENT_PROMPT_MAX_TOKENS:
            return f"The prompt is too long (about {prompt_token_count} tokens) while the limit is {SUBAGENT_PROMPT_MAX_TOKENS} tokens."

        subagent_type = subagent_types[subagent_type_id]
        client = clients.get_async(subagent_type_id)
        thread = await client.threads.create()
        run = await client.runs.create(
            thread_id=thread["thread_id"],
            assistant_id=subagent_type["assistant_id"],
            input={"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": SUBAGENT_RECURSION_LIMIT},
        )
        subagent_run_id = run["run_id"]
        subagent_run: SubagentRun = {
            "subagent_run_id": subagent_run_id,
            "subagent_type_id": subagent_type_id,
            "assistant_id": subagent_type["assistant_id"],
            "thread_id": thread["thread_id"],
            "run_id": run["run_id"],
            "status": run["status"],
            "prompt": prompt,
        }
        tool_output: dict[str, Any] = {
            "subagent_run_id": subagent_run_id,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(tool_output, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "subagent_runs": {subagent_run_id: subagent_run},
            }
        )

    return StructuredTool.from_function(
        func=spawn_subagent,
        coroutine=aspawn_subagent,
        name="spawn_subagent",
        description=_build_spawn_subagent_tool_description(subagent_types),
        infer_schema=False,
        args_schema=_build_spawn_subagent_schema(subagent_types),
    )


def _build_wait_tool(
    clients: _ClientCache,
) -> StructuredTool:

    def wait(runtime: ToolRuntime) -> str | Command:
        subagent_runs: dict[str, SubagentRun] = runtime.state.get("subagent_runs") or {}
        current_runs = {k: v for k, v in subagent_runs.items() if v["status"] not in TERMINAL_STATUSES}
        if not current_runs:
            return "No running subagents to wait for."

        deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            tool_output: list[dict[str, Any]] = []
            updated_runs: dict[str, SubagentRun] = {}

            for subagent_run_id, subagent_run in current_runs.items():
                client = clients.get_sync(subagent_run["subagent_type_id"])
                run = client.runs.get(thread_id=subagent_run["thread_id"], run_id=subagent_run["run_id"])

                if run["status"] not in TERMINAL_STATUSES:
                    if run["status"] != subagent_run["status"]:
                        updated_runs[subagent_run_id] = {
                            **subagent_run,
                            "status": run["status"],
                        }
                    continue

                tool_call_count = None
                content = None

                if run["status"] == "success":
                    history = client.threads.get_history(
                        thread_id=run["thread_id"],
                        limit=HISTORY_LIMIT,
                        metadata={"run_id": run["run_id"]},
                    )
                    if not history:
                        raise ValueError(f"No history found for run `{run['run_id']}`.")
                    if history[-1]["metadata"]["source"] != "input":
                        raise ValueError(f"History is truncated; increase `HISTORY_LIMIT`.")

                    before_messages = history[-1]["values"]["messages"]
                    after_messages = history[0]["values"]["messages"]
                    run_messages = after_messages[len(before_messages):]
                    if not run_messages:
                        raise ValueError(f"No messages found for run `{run['run_id']}`.")

                    tool_call_count = 0
                    for message in run_messages:
                        if message["type"] == "ai":
                            tool_call_count += len(message.get("tool_calls") or [])

                    last_message = run_messages[-1]
                    if last_message["type"] == "ai" and not last_message.get("tool_calls"):
                        content = last_message.get("content")
                        if content is not None and not isinstance(content, str):
                            content = json.dumps(content, ensure_ascii=False)

                elif run["status"] == "error":
                    error = run.get("error")
                    content = str(error) if error else None

                content, _ = _truncate_text(content, SUBAGENT_REPORT_MAX_TOKENS)

                report: SubagentReport = {
                    "subagent_run_id": subagent_run_id,
                    "status": run["status"],
                    "content": content,
                }
                tool_output.append(report)
                updated_runs[subagent_run_id] = {
                    **subagent_run,
                    "status": run["status"],
                    "tool_call_count": tool_call_count,
                    "report": report,
                }

            if tool_output:
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                json.dumps(tool_output, ensure_ascii=False),
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "subagent_runs": updated_runs,
                    }
                )

            time.sleep(SYNC_WAIT_POLL_SECONDS)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"No current subagent runs completed.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "subagent_runs": updated_runs,
            }
        )

    async def await_(runtime: ToolRuntime) -> str | Command:
        subagent_runs: dict[str, SubagentRun] = runtime.state.get("subagent_runs") or {}
        current_runs = {k: v for k, v in subagent_runs.items() if v["status"] not in TERMINAL_STATUSES}
        if not current_runs:
            return "No running subagents to wait for."

        async def get_run_items(subagent_run_id: str, subagent_run: SubagentRun):
            client = clients.get_async(subagent_run["subagent_type_id"])
            run = await client.runs.get(thread_id=subagent_run["thread_id"], run_id=subagent_run["run_id"])
            return subagent_run_id, subagent_run, run

        run_items = await asyncio.gather(*(get_run_items(k, v) for k, v in current_runs.items()))
        updated_runs: dict[str, SubagentRun] = {}
        finished_run_items = []
        for subagent_run_id, subagent_run, run in run_items:
            if run["status"] in TERMINAL_STATUSES:
                finished_run_items.append((subagent_run_id, subagent_run, run))
            elif run["status"] != subagent_run["status"]:
                updated_runs[subagent_run_id] = {
                    **subagent_run,
                    "status": run["status"],
                }

        if not finished_run_items:
            async def join_run_items(subagent_run_id: str, subagent_run: SubagentRun):
                client = clients.get_async(subagent_run["subagent_type_id"])
                await client.runs.join(thread_id=subagent_run["thread_id"], run_id=subagent_run["run_id"])
                run = await client.runs.get(thread_id=subagent_run["thread_id"], run_id=subagent_run["run_id"])
                return subagent_run_id, subagent_run, run

            tasks = [asyncio.create_task(join_run_items(k, v)) for k, v in current_runs.items()]
            done, pending = await asyncio.wait(
                tasks,
                timeout=WAIT_TIMEOUT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()

            if not done:
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                "No current subagent runs completed.",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "subagent_runs": updated_runs,
                    }
                )

            run_items = await asyncio.gather(*(get_run_items(k, v) for k, v in current_runs.items()))
            finished_run_items = []
            for subagent_run_id, subagent_run, run in run_items:
                if run["status"] in TERMINAL_STATUSES:
                    finished_run_items.append((subagent_run_id, subagent_run, run))
                elif run["status"] != subagent_run["status"]:
                    updated_runs[subagent_run_id] = {
                        **subagent_run,
                        "status": run["status"],
                    }

        if not finished_run_items:
            raise ValueError("No subagent runs with terminal status found despite "
                             "`client.runs.join` completing for some of them.")

        tool_output: list[dict[str, Any]] = []
        for subagent_run_id, subagent_run, run in finished_run_items:
            tool_call_count = None
            content = None

            if run["status"] == "success":
                client = clients.get_async(subagent_run["subagent_type_id"])
                history = await client.threads.get_history(
                    thread_id=run["thread_id"],
                    limit=HISTORY_LIMIT,
                    metadata={"run_id": run["run_id"]},
                )
                if not history:
                    raise ValueError(f"No history found for run `{run['run_id']}`.")
                if history[-1]["metadata"]["source"] != "input":
                    raise ValueError(f"History is truncated; increase `HISTORY_LIMIT`.")

                before_messages = history[-1]["values"]["messages"]
                after_messages = history[0]["values"]["messages"]
                run_messages = after_messages[len(before_messages):]
                if not run_messages:
                    raise ValueError(f"No messages found for run `{run['run_id']}`.")

                tool_call_count = 0
                for message in run_messages:
                    if message["type"] == "ai":
                        tool_call_count += len(message.get("tool_calls") or [])

                last_message = run_messages[-1]
                if last_message["type"] == "ai" and not last_message.get("tool_calls"):
                    content = last_message.get("content")
                    if content is not None and not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)

            elif run["status"] == "error":
                error = run.get("error")
                content = str(error) if error else None

            content, _ = _truncate_text(content, SUBAGENT_REPORT_MAX_TOKENS)

            report: SubagentReport = {
                "subagent_run_id": subagent_run_id,
                "status": run["status"],
                "content": content,
            }
            tool_output.append(report)
            updated_runs[subagent_run_id] = {
                **subagent_run,
                "status": run["status"],
                "tool_call_count": tool_call_count,
                "report": report,
            }

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(tool_output, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "subagent_runs": updated_runs,
            }
        )

    return StructuredTool.from_function(
        func=wait,
        coroutine=await_,
        name="wait",
        description=WAIT_TOOL_DESCRIPTION,
    )


def _build_decomposer_agent_tools(
    subagent_types: dict[str, SubagentType],
) -> list[StructuredTool]:
    clients = _ClientCache(subagent_types)
    return [
        _build_spawn_subagent_tool(subagent_types, clients),
        _build_wait_tool(clients),
    ]


class DecomposerAgentMiddleware(AgentMiddleware[DecomposerAgentState, ContextT, ResponseT]):
    state_schema = DecomposerAgentState

    def __init__(
        self,
        subagent_types: Sequence[SubagentType],
    ) -> None:
        super().__init__()

        if not subagent_types:
            msg = "At least one subagent must be specified"
            raise ValueError(msg)

        ids = [a["subagent_type_id"] for a in subagent_types]
        dupes = {id for id in ids if ids.count(id) > 1}
        if dupes:
            msg = f"Duplicate subagent type IDs: {dupes}"
            raise ValueError(msg)

        subagent_types = {a["subagent_type_id"]: a for a in subagent_types}
        self.tools = _build_decomposer_agent_tools(subagent_types)


def create_decomposer_agent(
    decomposer_model: str | BaseChatModel,
    subagent_types: Sequence[SubagentType],
    *,
    checkpointer: Checkpointer | None = None,
) -> CompiledStateGraph:
    """
    Create a Decomposer agent.

    Given a user prompt, the Decomposer agent starts spawning subagents and
    prompting them to achieve the goal. Depending on the subagents' reports,
    the Decomposer agent either spawns more subagents or responds to the user.

    The Decomposer agent works in a standard tool-calling loop. At each step, it
    can call one of the following tools:
    - `spawn_subagent` with args `subagent_type_id` and `prompt`. This tool
      spawns a new subagent of a certain type, asynchronously runs it on the
      given prompt, and immediately returns the subagent run ID. It does not
      return the subagent's report.
    - `wait` with no args. This tool waits for at least one new subagent report
      and returns all new subagent reports that have been produced since the
      last `wait` call. If there are no new reports and no running subagents,
      it returns "No running subagents to wait for."

    Args:
        decomposer_model: The language model for the Decomposer agent.
        subagent_types: The available subagent types.
        checkpointer: The checkpointer for the Decomposer agent.
    """
    middleware = [
        DecomposerAgentMiddleware(subagent_types),
    ]
    return create_agent(
        model=decomposer_model,
        tools=[],
        system_prompt=DECOMPOSER_SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=checkpointer,
    )
