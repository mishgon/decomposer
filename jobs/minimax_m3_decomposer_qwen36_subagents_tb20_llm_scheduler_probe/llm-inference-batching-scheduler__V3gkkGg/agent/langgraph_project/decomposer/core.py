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
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Command
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


DECOMPOSER_SYSTEM_PROMPT = f"""You are a Decomposer agent.

You initially see only the user prompt specifying your goal in the environment.
You can interact with the environment only via spawning and prompting subagents
of different types and reading their reports. You can decompose your goal into
immediate subgoals, prompt subagents to achieve them, read their reports, define
next subgoals, prompt new subagents, etc., until you are sure that your main
goal has been achieved.

Specifically, you can interact with subagents by calling the following tools:
- `spawn_subagent` with `prompt` and `subagent_type_id` args. It spawns a fresh
  subagent and runs it with the given prompt asynchronously in the background.
  You can imagine that the spawned subagents have their own isolated contexts
  and tools for interacting with the environment, do their best to follow your
  prompts, and finally write and submit their reports to a shared queue. If a
  subagent fails with an error, it also submits a report with the error message.
- `wait` with no args. It waits for at least one subagent report to be available
  in the queue and dequeues all reports from the queue as the tool output.

Note that each report contains only the subagent's final message, while the tool
calls and the tool outputs are not included by default. Thus, your and
subagents' contexts are isolated from each other by design. This is intentional
to prevent context rot and overflow. However, you can explicitly ask a subagent
to include any information you really need in its final response.

Follow these general principles:
- If you are uncertain about how to prompt the next subagent toward the goal (or
  a subgoal), identify what *minimal* information is needed to resolve your
  uncertainty, and spawn new read-only subagents or wait for already spawned
  subagents to gather the missing information. Proceed to spawn and prompt new
  subagents toward the initial goal (subgoal) when you are certain that they
  will make progress.
- The word "minimal" in the previous bullet point means that you should avoid
  asking subagents to report raw tool outputs, broad environment dumps, or other
  redundant information. Explicitly instruct subagents what to include and what
  not to include in their final message, which constitutes the report's content.
- Do not assume a subagent succeeded until it is claimed in the `content`
  field of its report. Note that the `status` field only reflects whether the
  subagent has completed without errors or interruptions, but does not reflect
  whether the subagent solved the prompted task properly.
- Before assigning a task to a subagent, make sure it cannot be split into
  smaller subtasks that can be *safely* executed in parallel. Otherwise, assign
  the parallel subtasks to multiple concurrent subagents.
- Spawn concurrent subagents only when you are 100% certain that they will not
  interfere with each other. Otherwise, wait for the first subagent to complete
  and spawn the next subagent.
- When `wait` returns "No running subagents to wait for.", spawn new subagents;
  do not keep waiting.
- Respond to the user only when the subagents' reports provide sufficient
  evidence that the goal has been achieved, or explain the blocker.
"""


SPAWN_SUBAGENT_TOOL_DESCRIPTION_TEMPLATE = """Spawns a fresh subagent of a
certain type and runs it in the background with the given prompt.

Use this tool when you want to delegate a task to a fresh subagent of a certain
type (e.g., gather a piece of information or change the environmental state in
a specific way).

You specify the task in the `prompt` argument. In principle, it can be any
free-form text (e.g., question, query, instructions, etc.), but you should avoid
broad or vague prompts. Specify the subagent's goal and provide helpful
instructions on how to achieve it, if you can. Finally, describe in detail what
information subagent must include in its final response. Note that subagents
might not know that their context is isolated from yours and it is your
responsibility to instruct subagents clearly to produce an informative final
response.

You should avoid delegating risky tasks that may lead to harmful and
irreversible changes in the environment. To reduce your uncertainty, spawn
read-only subagents. If some subagent types are read-only by design, this will
be reflected in their descriptions. You can also ask any subagent to be
read-only by explicitly prompting it not to change the environment state.

Avoid asking subagents to be too verbose, e.g., to report tool-calling traces,
broad environment dumps, or other redundant information.

Depending on the task, you should spawn a subagent of an appropriate type.
Specify the type using the `subagent_type_id` argument. Available subagent
types are listed in the table below:

| Agent type ID | Description |
| --- | --- |
{available_subagent_types}

When called, this tool creates a new subagent with a fresh context,
asynchronously runs it in the background, and returns immediately with
`subagent_run_id`, a unique identifier for the subagent run.

IMPORTANT: this tool does not return the subagent's report. Use `wait` to
collect subagent reports.
"""

WAIT_TIMEOUT_SECONDS = 60.0
WAIT_TOOL_DESCRIPTION = f"""Waits for at least one new report to become available
and returns all new subagent reports that have been produced since the last
`wait` call.

Use this tool when you have already delegated all tasks you find necessary at
the moment, the subagents are not done yet, and you want to wait for updates.

This tool takes no arguments. If there are no new reports and no running
subagents, it returns immediately with "No running subagents to wait for."
If any subagents have completed since the last `wait` call, it immediately
returns their reports. Otherwise, it waits for {WAIT_TIMEOUT_SECONDS} seconds
until at least one running subagent completes and returns its report. On
timeout, it returns "No current subagent runs completed."

The reports are formatted as a JSON list. Each report contains
`subagent_run_id`, `status`, `content`, and `error_message` fields. Use the
`subagent_run_id` field to identify the subagent run that produced the report.
Note that the `status` field only reflects whether the subagent has completed
without errors or interruptions, but does not reflect whether the subagent
solved the prompted task properly. If `status` is `"success"`, the `content`
field contains the subagent's final message. If it is empty, this means that
your prompt did not instruct the subagent clearly enough to return a final
response. If `status` is `"error"`, the `error_message` field contains the error
message, if any.

IMPORTANT: do not use this tool when there are no running subagents to wait for.
If you call it once and it returns "No running subagents to wait for.", do not
call it again until you have spawned new subagents.
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
    error_message: str | None


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


class SpawnSubagentSchema(BaseModel):
    prompt: str = Field(
        description="The free-form text prompt to send to the spawned subagent."
    )
    subagent_type_id: str = Field(
        description="The ID of the subagent type to spawn. Must be one of the available subagent type IDs."
    )


class DecomposerAgentState(AgentState[ResponseT]):
    subagent_runs: Annotated[NotRequired[dict[str, SubagentRun]], _subagent_runs_reducer]


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
        available_subagent_types=subagent_types_desc
    )


def _build_spawn_subagent_tool(
    subagent_types: dict[str, SubagentType],
    clients: _ClientCache,
) -> StructuredTool:

    def spawn_subagent(
        prompt: str,
        subagent_type_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type_id not in subagent_types:
            allowed = ", ".join(f"`{k}`" for k in subagent_types)
            return f"Unknown subagent type ID `{subagent_type_id}`. Available IDs: {allowed}."

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
        prompt: str,
        subagent_type_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type_id not in subagent_types:
            allowed = ", ".join(f"`{k}`" for k in subagent_types)
            return f"Unknown subagent type ID `{subagent_type_id}`. Available IDs: {allowed}."

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
        args_schema=SpawnSubagentSchema,
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
                error_message = None

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
                        if not isinstance(content, str) and content is not None:
                            content = json.dumps(content, ensure_ascii=False)

                elif run["status"] == "error":
                    error = run.get("error")
                    error_message = str(error) if error else None

                report: SubagentReport = {
                    "subagent_run_id": subagent_run_id,
                    "status": run["status"],
                    "content": content,
                    "error_message": error_message,
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
            error_message = None

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
                    if not isinstance(content, str) and content is not None:
                        content = json.dumps(content, ensure_ascii=False)

            elif run["status"] == "error":
                error = run.get("error")
                error_message = str(error) if error else None

            report: SubagentReport = {
                "subagent_run_id": subagent_run_id,
                "status": run["status"],
                "content": content,
                "error_message": error_message,
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
    - `spawn_subagent` with args `prompt` and `subagent_type_id`. This tool
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
