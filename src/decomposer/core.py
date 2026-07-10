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

from .prompts import (
    DECOMPOSER_SYSTEM_PROMPT, SPAWN_SUBAGENT_TOOL_DESCRIPTION,
    PROMPT_PARAMETER_DESCRIPTION, SUBAGENT_TYPE_ID_PARAMETER_DESCRIPTION,
    WAIT_TOOL_DESCRIPTION
)

logger = logging.getLogger(__name__)


SUBAGENT_PROMPT_MAX_TOKENS = 1024
SUBAGENT_REPORT_MAX_TOKENS = 1024
WAIT_TIMEOUT_SECONDS = 60.0
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


def _get_current_subagent_runs(
    subagent_runs: dict[str, SubagentRun],
) -> dict[str, SubagentRun]:
    terminal_runs_without_reports = {
        subagent_run_id: subagent_run
        for subagent_run_id, subagent_run in subagent_runs.items()
        if subagent_run["status"] in TERMINAL_STATUSES
        and subagent_run.get("report") is None
    }
    if terminal_runs_without_reports:
        details = ", ".join(
            f"`{subagent_run_id}` ({subagent_run['status']})"
            for subagent_run_id, subagent_run in terminal_runs_without_reports.items()
        )
        raise RuntimeError(
            "Invalid Decomposer state: terminal subagent runs have no collected "
            f"report: {details}."
        )

    return {
        subagent_run_id: subagent_run
        for subagent_run_id, subagent_run in subagent_runs.items()
        if subagent_run["status"] not in TERMINAL_STATUSES
    }


def _build_spawn_subagent_schema(subagent_types: dict[str, SubagentType]) -> type[BaseModel]:
    available_subagent_type_ids = ", ".join(f"`{k}`" for k in subagent_types)

    class SpawnSubagentSchema(BaseModel):
        subagent_type_id: str = Field(
            description=SUBAGENT_TYPE_ID_PARAMETER_DESCRIPTION.format(available_subagent_type_ids=available_subagent_type_ids)
        )
        prompt: str = Field(
            description=PROMPT_PARAMETER_DESCRIPTION.format(subagent_prompt_max_tokens=SUBAGENT_PROMPT_MAX_TOKENS)
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
    return SPAWN_SUBAGENT_TOOL_DESCRIPTION.format(available_subagent_types=subagent_types_desc)


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
        if run["status"] in TERMINAL_STATUSES:
            raise ValueError(f"`client.runs.create` returned a run `{subagent_run_id}` with terminal status `{run['status']}`.")
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
        if run["status"] in TERMINAL_STATUSES:
            raise ValueError(f"`client.runs.create` returned a run `{subagent_run_id}` with terminal status `{run['status']}`.")
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
        current_runs = _get_current_subagent_runs(subagent_runs)
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
                        raise ValueError("History is truncated; increase `HISTORY_LIMIT`.")

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
                        "No current subagent runs completed.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "subagent_runs": updated_runs,
            }
        )

    async def await_(runtime: ToolRuntime) -> str | Command:
        subagent_runs: dict[str, SubagentRun] = runtime.state.get("subagent_runs") or {}
        current_runs = _get_current_subagent_runs(subagent_runs)
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
                    raise ValueError("History is truncated; increase `HISTORY_LIMIT`.")

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
        description=WAIT_TOOL_DESCRIPTION.format(wait_timeout_seconds=WAIT_TIMEOUT_SECONDS,
                                                 subagent_report_max_tokens=SUBAGENT_REPORT_MAX_TOKENS),
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
    middleware: Sequence[AgentMiddleware] | None = None,
    context_schema: type[Any] | None = None,
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
        middleware: Additional LangChain middleware for the Decomposer agent.
        context_schema: Runtime context schema passed through to `create_agent`.
    """
    system_prompt = DECOMPOSER_SYSTEM_PROMPT.format(subagent_report_max_tokens=SUBAGENT_REPORT_MAX_TOKENS)
    agent_middleware = [
        DecomposerAgentMiddleware(subagent_types),
        *(middleware or []),
    ]
    return create_agent(
        model=decomposer_model,
        tools=[],
        system_prompt=system_prompt,
        middleware=agent_middleware,
        checkpointer=checkpointer,
        context_schema=context_schema,
    )
