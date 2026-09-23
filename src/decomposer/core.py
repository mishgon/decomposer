import json
import time
import asyncio
import logging
from collections import Counter
from typing import Annotated, Any, NotRequired, Sequence

from httpx import ReadError, RemoteProtocolError
from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, Command
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .prompts import (
    NEW_TOOL_DESCRIPTION,
    FORK_TOOL_DESCRIPTION,
    PARALLEL_FORK_RUN_CALL_ERROR,
    DECOMPOSER_SYSTEM_PROMPT,
    PARALLEL_RUN_CALL_ERROR,
    EARLY_RESPONSE_ERROR,
    EMPTY_RESPONSE_ERROR,
    FAILED_RUN_ERROR,
    WAIT_TIMEOUT_ERROR,
    NO_ACTIVE_RUNS_ERROR,
    ACTIVE_RUN_ERROR,
    PARALLEL_WAIT_CALL_ERROR,
    PROMPT_PARAMETER_DESCRIPTION,
    RUN_TOOL_DESCRIPTION,
    SUBAGENT_ID_PARAMETER_DESCRIPTION,
    SUBAGENT_TYPE_ID_PARAMETER_DESCRIPTION,
    UNKNOWN_SUBAGENT_ERROR,
    UNKNOWN_SUBAGENT_TYPE_ERROR,
    WAIT_TOOL_DESCRIPTION,
)

logger = logging.getLogger(__name__)


# SUBAGENT_PROMPT_MAX_TOKENS = 1024
# SUBAGENT_RESPONSE_MAX_TOKENS = 1024
WAIT_TIMEOUT_SECONDS = 60.0
WAIT_POLL_SECONDS = 5.0
TERMINAL_STATUSES = frozenset({"responded", "error", "timeout", "interrupted"})
HISTORY_LIMIT = 1000


class SubagentType(TypedDict):
    subagent_type_id: str
    description: str
    assistant_id: str
    url: NotRequired[str]
    headers: NotRequired[dict[str, str]]


class SubagentToolCall(TypedDict):
    id: str
    name: str
    args: dict[str, Any]


class Subagent(TypedDict):
    subagent_id: str
    subagent_type_id: str
    assistant_id: str
    thread_id: str


class SubagentRun(TypedDict):
    subagent_run_id: str
    subagent_id: str
    run_id: str
    status: str
    prompt: str
    tool_calls: NotRequired[list[SubagentToolCall]]
    response: NotRequired[str | None]
    # Zero-based order in which wait() returned this run's response.
    response_sequence_number: NotRequired[int]
    error: NotRequired[str | None]


def _subagents_reducer(
    existing: dict[str, Subagent] | None,
    update: dict[str, Subagent],
) -> dict[str, Subagent]:
    merged = dict(existing or {})
    merged.update(update)
    return merged


def _subagent_runs_reducer(
    existing: dict[str, SubagentRun] | None,
    update: dict[str, SubagentRun],
) -> dict[str, SubagentRun]:
    merged = dict(existing or {})
    merged.update(update)
    return merged


def _normalize_run_status(status: str) -> str:
    return "responded" if status == "success" else status


def _get_current_subagent_runs(
    subagent_runs: dict[str, SubagentRun],
) -> dict[str, SubagentRun]:
    terminal_runs_without_responses = {
        subagent_run_id: subagent_run
        for subagent_run_id, subagent_run in subagent_runs.items()
        if subagent_run["status"] in TERMINAL_STATUSES
        and "response_sequence_number" not in subagent_run
    }
    if terminal_runs_without_responses:
        details = ", ".join(
            f"`{subagent_run_id}` ({subagent_run['status']})"
            for subagent_run_id, subagent_run in terminal_runs_without_responses.items()
        )
        raise RuntimeError(
            "Invalid Decomposer state: terminal subagent runs have no collected "
            f"response: {details}."
        )

    return {
        subagent_run_id: subagent_run
        for subagent_run_id, subagent_run in subagent_runs.items()
        if subagent_run["status"] not in TERMINAL_STATUSES
    }


def _build_new_schema(
    subagent_types: dict[str, SubagentType],
) -> type[BaseModel]:
    available_subagent_types = "\n".join(
        f"| {json.dumps(subagent_type_id, ensure_ascii=False)} | {subagent_type['description']} |"
        for subagent_type_id, subagent_type in subagent_types.items()
    )
    description = SUBAGENT_TYPE_ID_PARAMETER_DESCRIPTION.format(
        available_subagent_types=available_subagent_types
    )

    class NewSchema(BaseModel):
        subagent_type_id: str = Field(description=description)

    return NewSchema


class ForkSchema(BaseModel):
    subagent_id: str = Field(description=SUBAGENT_ID_PARAMETER_DESCRIPTION)


class RunSchema(BaseModel):
    subagent_id: str = Field(description=SUBAGENT_ID_PARAMETER_DESCRIPTION)
    prompt: str = Field(description=PROMPT_PARAMETER_DESCRIPTION)


class DecomposerAgentState(AgentState[ResponseT]):
    subagents: Annotated[
        NotRequired[dict[str, Subagent]], _subagents_reducer
    ]
    subagent_runs: Annotated[
        NotRequired[dict[str, SubagentRun]], _subagent_runs_reducer
    ]


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


def _extract_subagent_tool_calls(run_messages: list[dict[str, Any]]) -> list[SubagentToolCall]:
    tool_calls: list[SubagentToolCall] = []
    for message in run_messages:
        if message["type"] != "ai":
            continue
        for tool_call in message["tool_calls"]:
            tool_call_id = tool_call["id"]
            name = tool_call["name"]
            args = tool_call["args"]
            if not isinstance(tool_call_id, str) or not isinstance(name, str) or not isinstance(args, dict):
                raise ValueError(f"Invalid subagent tool call in run history: {tool_call!r}")
            tool_calls.append({"id": tool_call_id, "name": name, "args": args})
    return tool_calls


def _resolve_headers(subagent_type: SubagentType) -> dict[str, str]:
    headers: dict[str, str] = dict(subagent_type.get("headers") or {})
    if "x-auth-scheme" not in headers:
        headers["x-auth-scheme"] = "langsmith"
    return headers


class _ClientCache:
    """Adapted from deepagents.middleware.async_subagents.ClientCache."""

    def __init__(self, subagent_types: dict[str, SubagentType]) -> None:
        self._subagent_types = subagent_types
        self._sync: dict[
            tuple[str | None, frozenset[tuple[str, str]]], SyncLangGraphClient
        ] = {}
        self._async: dict[
            tuple[str | None, frozenset[tuple[str, str]]], LangGraphClient
        ] = {}

    def _cache_key(
        self, subagent_type: SubagentType
    ) -> tuple[str | None, frozenset[tuple[str, str]]]:
        return (
            subagent_type.get("url"),
            frozenset(_resolve_headers(subagent_type).items()),
        )

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


def _build_new_tool(
    subagent_types: dict[str, SubagentType],
    clients: _ClientCache,
) -> StructuredTool:

    def new(
        subagent_type_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type_id not in subagent_types:
            allowed = ", ".join(f"`{k}`" for k in subagent_types)
            return UNKNOWN_SUBAGENT_TYPE_ERROR.format(
                subagent_type_id=subagent_type_id, allowed=allowed
            )

        subagent_type = subagent_types[subagent_type_id]
        client = clients.get_sync(subagent_type_id)
        thread = client.threads.create()
        subagent_id = thread["thread_id"]
        subagent: Subagent = {
            "subagent_id": subagent_id,
            "subagent_type_id": subagent_type_id,
            "assistant_id": subagent_type["assistant_id"],
            "thread_id": thread["thread_id"],
        }
        tool_output: dict[str, Any] = {
            "subagent_id": subagent_id,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(tool_output, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "subagents": {subagent_id: subagent},
            }
        )

    async def anew(
        subagent_type_id: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type_id not in subagent_types:
            allowed = ", ".join(f"`{k}`" for k in subagent_types)
            return UNKNOWN_SUBAGENT_TYPE_ERROR.format(
                subagent_type_id=subagent_type_id, allowed=allowed
            )

        subagent_type = subagent_types[subagent_type_id]
        client = clients.get_async(subagent_type_id)
        thread = await client.threads.create()
        subagent_id = thread["thread_id"]
        subagent: Subagent = {
            "subagent_id": subagent_id,
            "subagent_type_id": subagent_type_id,
            "assistant_id": subagent_type["assistant_id"],
            "thread_id": thread["thread_id"],
        }
        tool_output: dict[str, Any] = {
            "subagent_id": subagent_id,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(tool_output, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "subagents": {subagent_id: subagent},
            }
        )

    return StructuredTool.from_function(
        func=new,
        coroutine=anew,
        name="new",
        description=NEW_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=_build_new_schema(subagent_types),
    )


def _get_subagent_error(
    subagent_id: str,
    state: DecomposerAgentState,
) -> str | None:
    if subagent_id not in state["subagents"]:
        return UNKNOWN_SUBAGENT_ERROR.format(subagent_id=subagent_id)

    for subagent_run_id, subagent_run in state["subagent_runs"].items():
        if subagent_run["subagent_id"] != subagent_id:
            continue
        if "response_sequence_number" not in subagent_run:
            return ACTIVE_RUN_ERROR.format(
                subagent_id=subagent_id, subagent_run_id=subagent_run_id
            )
        if subagent_run["status"] != "responded":
            return FAILED_RUN_ERROR.format(
                subagent_id=subagent_id,
                subagent_run_id=subagent_run_id,
                status=subagent_run["status"],
            )
    return None


def _build_fork_tool(clients: _ClientCache) -> StructuredTool:

    def fork(subagent_id: str, runtime: ToolRuntime) -> str | Command:
        error = _get_subagent_error(subagent_id, runtime.state)
        if error is not None:
            return error

        source = runtime.state["subagents"][subagent_id]
        client = clients.get_sync(source["subagent_type_id"])
        thread = client.threads.copy(source["thread_id"])
        subagent_id = thread["thread_id"]
        subagent: Subagent = {
            **source,
            "subagent_id": subagent_id,
            "thread_id": thread["thread_id"],
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps({"subagent_id": subagent_id}, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "subagents": {subagent_id: subagent},
            }
        )

    async def afork(subagent_id: str, runtime: ToolRuntime) -> str | Command:
        error = _get_subagent_error(subagent_id, runtime.state)
        if error is not None:
            return error

        source = runtime.state["subagents"][subagent_id]
        client = clients.get_async(source["subagent_type_id"])
        thread = await client.threads.copy(source["thread_id"])
        subagent_id = thread["thread_id"]
        subagent: Subagent = {
            **source,
            "subagent_id": subagent_id,
            "thread_id": thread["thread_id"],
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps({"subagent_id": subagent_id}, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "subagents": {subagent_id: subagent},
            }
        )

    return StructuredTool.from_function(
        func=fork,
        coroutine=afork,
        name="fork",
        description=FORK_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=ForkSchema,
    )


def _build_run_tool(
    clients: _ClientCache,
    recursion_limit: int | None,
) -> StructuredTool:

    def run(
        subagent_id: str,
        prompt: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _get_subagent_error(subagent_id, runtime.state)
        if error is not None:
            return error

        # prompt_token_count = _count_text_tokens(prompt)
        # if prompt_token_count > SUBAGENT_PROMPT_MAX_TOKENS:
        #     return f"The prompt is too long (about {prompt_token_count} tokens) while the limit is {SUBAGENT_PROMPT_MAX_TOKENS} tokens."

        subagent = runtime.state["subagents"][subagent_id]
        client = clients.get_sync(subagent["subagent_type_id"])
        run = client.runs.create(
            thread_id=subagent["thread_id"],
            assistant_id=subagent["assistant_id"],
            input={"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": recursion_limit} if recursion_limit is not None else None,
            context=runtime.context,
            multitask_strategy="reject",
        )

        subagent_run_id = run["run_id"]
        status = _normalize_run_status(run["status"])
        if status in TERMINAL_STATUSES:
            raise ValueError(
                f"`client.runs.create` returned a run `{subagent_run_id}` with terminal status `{status}`."
            )
        subagent_run: SubagentRun = {
            "subagent_run_id": subagent_run_id,
            "subagent_id": subagent_id,
            "run_id": run["run_id"],
            "status": status,
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

    async def arun(
        subagent_id: str,
        prompt: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _get_subagent_error(subagent_id, runtime.state)
        if error is not None:
            return error

        # prompt_token_count = _count_text_tokens(prompt)
        # if prompt_token_count > SUBAGENT_PROMPT_MAX_TOKENS:
        #     return f"The prompt is too long (about {prompt_token_count} tokens) while the limit is {SUBAGENT_PROMPT_MAX_TOKENS} tokens."

        subagent = runtime.state["subagents"][subagent_id]
        client = clients.get_async(subagent["subagent_type_id"])
        run = await client.runs.create(
            thread_id=subagent["thread_id"],
            assistant_id=subagent["assistant_id"],
            input={"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": recursion_limit} if recursion_limit is not None else None,
            context=runtime.context,
            multitask_strategy="reject",
        )

        subagent_run_id = run["run_id"]
        status = _normalize_run_status(run["status"])
        if status in TERMINAL_STATUSES:
            raise ValueError(
                f"`client.runs.create` returned a run `{subagent_run_id}` with terminal status `{status}`."
            )
        subagent_run: SubagentRun = {
            "subagent_run_id": subagent_run_id,
            "subagent_id": subagent_id,
            "run_id": run["run_id"],
            "status": status,
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
        func=run,
        coroutine=arun,
        name="run",
        description=RUN_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=RunSchema,
    )


def _build_wait_tool(
    clients: _ClientCache,
) -> StructuredTool:

    def wait(runtime: ToolRuntime) -> str | Command:
        subagents: dict[str, Subagent] = runtime.state["subagents"]
        subagent_runs: dict[str, SubagentRun] = runtime.state["subagent_runs"]
        current_runs = _get_current_subagent_runs(subagent_runs)
        if not current_runs:
            return NO_ACTIVE_RUNS_ERROR
        next_response_sequence_number = len(subagent_runs) - len(current_runs)

        deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
        while True:
            tool_output: list[dict[str, Any]] = []
            updated_runs: dict[str, SubagentRun] = {}

            for subagent_run_id, subagent_run in current_runs.items():
                subagent = subagents[subagent_run["subagent_id"]]
                client = clients.get_sync(subagent["subagent_type_id"])
                run = client.runs.get(
                    thread_id=subagent["thread_id"],
                    run_id=subagent_run["run_id"],
                )

                status = _normalize_run_status(run["status"])
                if status not in TERMINAL_STATUSES:
                    if status != subagent_run["status"]:
                        updated_runs[subagent_run_id] = {
                            **subagent_run,
                            "status": status,
                        }
                    continue

                history = client.threads.get_history(
                    thread_id=run["thread_id"],
                    limit=HISTORY_LIMIT,
                    metadata={"run_id": run["run_id"]},
                )
                if history:
                    if history[-1]["metadata"]["source"] != "input":
                        raise ValueError(
                            "History is truncated; increase `HISTORY_LIMIT`."
                        )

                    before_messages = history[-1]["values"]["messages"]
                    after_messages = history[0]["values"]["messages"]
                    run_messages = after_messages[len(before_messages) :]
                else:
                    run_messages = []

                if status == "responded" and not run_messages:
                    raise ValueError(
                        f"No messages found for run `{run['run_id']}`."
                    )

                tool_calls = _extract_subagent_tool_calls(run_messages)

                response = None
                error = None
                if status == "responded":
                    last_message = run_messages[-1]
                    if (
                        last_message["type"] != "ai"
                        or last_message["tool_calls"]
                        or last_message["content"] is None
                    ):
                        raise ValueError(
                            "Expected a final AI message without tool calls and with non-null content "
                            f"for responded run `{run['run_id']}`."
                        )
                    response = last_message["content"]
                    if not isinstance(response, str):
                        response = json.dumps(response, ensure_ascii=False)
                elif status == "error":
                    thread = client.threads.get(thread_id=run["thread_id"])
                    error = thread.get("error")
                    error = json.dumps(error, ensure_ascii=False) if error else None
                # response, _ = _truncate_text(response, SUBAGENT_RESPONSE_MAX_TOKENS)

                response_sequence_number = next_response_sequence_number + len(tool_output)
                tool_output.append({
                    "subagent_id": subagent_run["subagent_id"],
                    "subagent_run_id": subagent_run_id,
                    "status": status,
                    "response": response,
                    "error": error,
                })
                updated_runs[subagent_run_id] = {
                    **subagent_run,
                    "status": status,
                    "tool_calls": tool_calls,
                    "response": response,
                    "error": error,
                    "response_sequence_number": response_sequence_number,
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

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                WAIT_TIMEOUT_ERROR,
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "subagent_runs": updated_runs,
                    }
                )

            time.sleep(min(WAIT_POLL_SECONDS, remaining))

    async def await_(runtime: ToolRuntime) -> str | Command:
        subagents: dict[str, Subagent] = runtime.state["subagents"]
        subagent_runs: dict[str, SubagentRun] = runtime.state["subagent_runs"]
        current_runs = _get_current_subagent_runs(subagent_runs)
        if not current_runs:
            return NO_ACTIVE_RUNS_ERROR
        next_response_sequence_number = len(subagent_runs) - len(current_runs)

        deadline = asyncio.get_running_loop().time() + WAIT_TIMEOUT_SECONDS
        while True:
            tool_output: list[dict[str, Any]] = []
            updated_runs: dict[str, SubagentRun] = {}

            runs = await asyncio.gather(
                *(
                    clients.get_async(
                        subagents[subagent_run["subagent_id"]]["subagent_type_id"]
                    ).runs.get(
                        thread_id=subagents[subagent_run["subagent_id"]]["thread_id"],
                        run_id=subagent_run["run_id"],
                    )
                    for subagent_run in current_runs.values()
                ),
                return_exceptions=True,
            )
            for (subagent_run_id, subagent_run), run in zip(
                current_runs.items(), runs, strict=True
            ):
                if isinstance(run, (ReadError, RemoteProtocolError)):
                    continue
                if isinstance(run, BaseException):
                    raise run

                status = _normalize_run_status(run["status"])
                if status not in TERMINAL_STATUSES:
                    if status != subagent_run["status"]:
                        updated_runs[subagent_run_id] = {
                            **subagent_run,
                            "status": status,
                        }
                    continue

                subagent = subagents[subagent_run["subagent_id"]]
                client = clients.get_async(subagent["subagent_type_id"])
                history = await client.threads.get_history(
                    thread_id=run["thread_id"],
                    limit=HISTORY_LIMIT,
                    metadata={"run_id": run["run_id"]},
                )
                if history:
                    if history[-1]["metadata"]["source"] != "input":
                        raise ValueError(
                            "History is truncated; increase `HISTORY_LIMIT`."
                        )

                    before_messages = history[-1]["values"]["messages"]
                    after_messages = history[0]["values"]["messages"]
                    run_messages = after_messages[len(before_messages) :]
                else:
                    run_messages = []

                if status == "responded" and not run_messages:
                    raise ValueError(f"No messages found for run `{run['run_id']}`.")

                tool_calls = _extract_subagent_tool_calls(run_messages)

                response = None
                error = None
                if status == "responded":
                    last_message = run_messages[-1]
                    if (
                        last_message["type"] != "ai"
                        or last_message["tool_calls"]
                        or last_message["content"] is None
                    ):
                        raise ValueError(
                            "Expected a final AI message without tool calls and with non-null content "
                            f"for responded run `{run['run_id']}`."
                        )
                    response = last_message["content"]
                    if not isinstance(response, str):
                        response = json.dumps(response, ensure_ascii=False)
                elif status == "error":
                    thread = await client.threads.get(thread_id=run["thread_id"])
                    error = thread.get("error")
                    error = json.dumps(error, ensure_ascii=False) if error else None
                # response, _ = _truncate_text(response, SUBAGENT_RESPONSE_MAX_TOKENS)

                response_sequence_number = next_response_sequence_number + len(tool_output)
                tool_output.append({
                    "subagent_id": subagent_run["subagent_id"],
                    "subagent_run_id": subagent_run_id,
                    "status": status,
                    "response": response,
                    "error": error,
                })
                updated_runs[subagent_run_id] = {
                    **subagent_run,
                    "status": status,
                    "tool_calls": tool_calls,
                    "response": response,
                    "error": error,
                    "response_sequence_number": response_sequence_number,
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

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                WAIT_TIMEOUT_ERROR,
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "subagent_runs": updated_runs,
                    }
                )

            await asyncio.sleep(min(WAIT_POLL_SECONDS, remaining))

    return StructuredTool.from_function(
        func=wait,
        coroutine=await_,
        name="wait",
        description=WAIT_TOOL_DESCRIPTION.format(
            wait_timeout_seconds=WAIT_TIMEOUT_SECONDS,
            no_active_runs_error=NO_ACTIVE_RUNS_ERROR,
            wait_timeout_error=WAIT_TIMEOUT_ERROR,
            # subagent_response_max_tokens=SUBAGENT_RESPONSE_MAX_TOKENS,
        ),
    )


def _build_decomposer_agent_tools(
    subagent_types: dict[str, SubagentType],
    subagent_recursion_limit: int | None,
) -> list[StructuredTool]:
    clients = _ClientCache(subagent_types)
    return [
        _build_new_tool(subagent_types, clients),
        _build_fork_tool(clients),
        _build_run_tool(clients, subagent_recursion_limit),
        _build_wait_tool(clients),
    ]


class DecomposerAgentMiddleware(AgentMiddleware[DecomposerAgentState, ContextT, ResponseT]):
    state_schema = DecomposerAgentState

    def __init__(
        self,
        subagent_types: Sequence[SubagentType],
        subagent_recursion_limit: int | None,
    ) -> None:
        super().__init__()

        if not subagent_types:
            msg = "At least one subagent must be specified"
            raise ValueError(msg)

        ids = [a["subagent_type_id"] for a in subagent_types]
        subagent_types = {a["subagent_type_id"]: a for a in subagent_types}
        if len(ids) > len(subagent_types):
            seen = set()
            dupes = {id for id in ids if id in seen or seen.add(id)}
            raise ValueError(f"Duplicate subagent type IDs: {dupes}")

        self.tools = _build_decomposer_agent_tools(
            subagent_types, subagent_recursion_limit
        )

    def after_model(
        self,
        state: DecomposerAgentState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        ai_message = next(
            (
                message
                for message in reversed(state["messages"])
                if isinstance(message, AIMessage)
            ),
            None,
        )
        if ai_message is None:
            raise RuntimeError("Expected the state to contain an AIMessage.")

        tool_calls = ai_message.tool_calls
        subagent_run_counts = Counter(
            tool_call["args"]["subagent_id"]
            for tool_call in tool_calls
            if tool_call["name"] == "run"
            and isinstance(tool_call["args"].get("subagent_id"), str)
        )
        forked_subagent_ids = {
            tool_call["args"]["subagent_id"]
            for tool_call in tool_calls
            if tool_call["name"] == "fork"
            and isinstance(tool_call["args"].get("subagent_id"), str)
        }
        rejected_calls = []
        for tool_call in tool_calls:
            if tool_call["name"] == "wait" and len(tool_calls) > 1:
                error = PARALLEL_WAIT_CALL_ERROR
            elif (
                tool_call["name"] in {"run", "fork"}
                and isinstance(tool_call["args"].get("subagent_id"), str)
                and tool_call["args"]["subagent_id"] in forked_subagent_ids
                and subagent_run_counts[tool_call["args"]["subagent_id"]] > 0
            ):
                error = PARALLEL_FORK_RUN_CALL_ERROR
            elif (
                tool_call["name"] == "run"
                and isinstance(tool_call["args"].get("subagent_id"), str)
                and subagent_run_counts[tool_call["args"]["subagent_id"]] > 1
            ):
                error = PARALLEL_RUN_CALL_ERROR
            else:
                continue
            rejected_calls.append(
                ToolMessage(
                    content=error,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        if rejected_calls:
            return {"messages": rejected_calls}

        if tool_calls:
            return None

        subagent_runs = state["subagent_runs"]
        if any("response_sequence_number" not in run for run in subagent_runs.values()):
            error = EARLY_RESPONSE_ERROR
        elif not ai_message.text.strip():
            error = EMPTY_RESPONSE_ERROR
        else:
            return None

        return {
            "messages": [HumanMessage(content=error)],
            "jump_to": "model",
        }


def create_decomposer_agent(
    *,
    decomposer_model: str | BaseChatModel,
    subagent_types: Sequence[SubagentType],
    middleware: Sequence[AgentMiddleware] | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    decomposer_recursion_limit: int | None = None,
    subagent_recursion_limit: int | None = None,
) -> CompiledStateGraph:
    decomposer_middelware = DecomposerAgentMiddleware(
        subagent_types, subagent_recursion_limit
    )

    agent = create_agent(
        model=decomposer_model,
        tools=[],
        system_prompt=DECOMPOSER_SYSTEM_PROMPT,
        middleware=[
            decomposer_middelware,
            *(middleware or []),
        ],
        context_schema=context_schema,
        checkpointer=checkpointer,
    )
    if decomposer_recursion_limit is not None:
        agent = agent.with_config(recursion_limit=decomposer_recursion_limit)
    return agent
