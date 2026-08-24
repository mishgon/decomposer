import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import decomposer.core as core
import pytest
from httpx import ReadError, RemoteProtocolError, Request
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from decomposer.core import (
    DecomposerAgentMiddleware,
    SubagentType,
    _build_spawn_subagent_tool,
    _build_wait_tool,
    _extract_subagent_tool_calls,
)
from decomposer.prompts import (
    EARLY_REPORT_ERROR,
    EMPTY_REPORT_ERROR,
    PARALLEL_WAIT_CALL_ERROR,
)


SUBAGENT_TYPE: SubagentType = {
    "subagent_type_id": "test",
    "description": "Test subagent",
    "assistant_id": "test_assistant",
    "url": "http://subagents.test",
}


def test_spawn_subagent_passes_context_to_sync_run() -> None:
    client = _SyncClient()
    tool = _build_spawn_subagent_tool(
        {"test": SUBAGENT_TYPE},
        _ClientCacheStub(sync_client=client),
        1,
    )

    assert tool.func is not None
    tool.func(
        subagent_type_id="test",
        prompt="do the task",
        runtime=SimpleNamespace(context={"value": 42}, tool_call_id="call_1"),
    )

    assert client.runs.create_kwargs["context"] == {"value": 42}


def test_spawn_subagent_passes_context_to_async_run() -> None:
    client = _AsyncClient()
    tool = _build_spawn_subagent_tool(
        {"test": SUBAGENT_TYPE},
        _ClientCacheStub(async_client=client),
        1,
    )

    assert tool.coroutine is not None
    asyncio.run(
        tool.coroutine(
            subagent_type_id="test",
            prompt="do the task",
            runtime=SimpleNamespace(context={"value": 42}, tool_call_id="call_1"),
        )
    )

    assert client.runs.create_kwargs["context"] == {"value": 42}


def test_spawn_subagent_rejects_prompt_over_limit() -> None:
    client = _SyncClient()
    tool = _build_spawn_subagent_tool(
        {"test": SUBAGENT_TYPE},
        _ClientCacheStub(sync_client=client),
        1,
    )

    assert tool.func is not None
    result = tool.func(
        subagent_type_id="test",
        prompt="word " * (core.SUBAGENT_PROMPT_MAX_TOKENS + 100),
        runtime=SimpleNamespace(context={}, tool_call_id="call_1"),
    )

    assert isinstance(result, str)
    assert f"limit is {core.SUBAGENT_PROMPT_MAX_TOKENS} tokens" in result
    assert client.runs.create_kwargs == {}


def test_extract_subagent_tool_calls_preserves_message_and_call_order() -> None:
    assert _extract_subagent_tool_calls(
        [
            {
                "type": "ai",
                "tool_calls": [
                    {"id": "call_1", "name": "first", "args": {"value": 1}},
                    {"id": "call_2", "name": "second", "args": {"value": 2}},
                ],
            },
            {"type": "tool", "tool_call_id": "call_2", "content": "result"},
            {
                "type": "ai",
                "tool_calls": [
                    {"id": "call_3", "name": "third", "args": {"value": 3}},
                ],
            },
        ]
    ) == [
        {"id": "call_1", "name": "first", "args": {"value": 1}},
        {"id": "call_2", "name": "second", "args": {"value": 2}},
        {"id": "call_3", "name": "third", "args": {"value": 3}},
    ]


def test_wait_stores_tool_calls_in_returned_report_order() -> None:
    client = _CompletedRunsClient()
    tool = _build_wait_tool(_ClientCacheStub(sync_client=client))
    runtime = SimpleNamespace(
        state={
            "subagent_runs": {
                "run_c": {
                    **_subagent_run("run_c"),
                    "status": "success",
                    "report": {
                        "subagent_run_id": "run_c",
                        "status": "success",
                        "content": "earlier report",
                    },
                    "report_sequence_number": 0,
                },
                "run_b": _subagent_run("run_b"),
                "run_a": _subagent_run("run_a"),
            }
        },
        tool_call_id="wait_1",
    )

    assert tool.func is not None
    command = tool.func(runtime=runtime)

    assert command.update["subagent_runs"]["run_b"]["report_sequence_number"] == 1
    assert command.update["subagent_runs"]["run_a"]["report_sequence_number"] == 2
    assert command.update["subagent_runs"]["run_b"]["tool_calls"] == [
        {"id": "run_b_call", "name": "resource_tool", "args": {"run": "run_b"}},
    ]
    assert command.update["subagent_runs"]["run_a"]["tool_calls"] == [
        {"id": "run_a_call", "name": "resource_tool", "args": {"run": "run_a"}},
    ]


def test_wait_stores_tool_calls_from_error_run() -> None:
    client = _CompletedRunsClient(status="error")
    tool = _build_wait_tool(_ClientCacheStub(sync_client=client))
    runtime = SimpleNamespace(
        state={"subagent_runs": {"run_a": _subagent_run("run_a")}},
        tool_call_id="wait_1",
    )

    assert tool.func is not None
    command = tool.func(runtime=runtime)

    subagent_run = command.update["subagent_runs"]["run_a"]
    assert subagent_run["tool_calls"] == [
        {"id": "run_a_call", "name": "resource_tool", "args": {"run": "run_a"}},
    ]
    assert subagent_run["report"]["content"] == "subagent failed"


def test_wait_truncates_long_report(monkeypatch) -> None:
    monkeypatch.setattr(core, "SUBAGENT_REPORT_MAX_TOKENS", 8)
    client = _CompletedRunsClient()
    history = client.threads.get_history(
        thread_id="run_a_thread",
        limit=core.HISTORY_LIMIT,
        metadata={"run_id": "run_a"},
    )
    history[0]["values"]["messages"][-1]["content"] = "word " * 100
    client.threads = SimpleNamespace(get_history=lambda **kwargs: history)
    tool = _build_wait_tool(_ClientCacheStub(sync_client=client))

    assert tool.func is not None
    command = tool.func(
        runtime=SimpleNamespace(
            state={"subagent_runs": {"run_a": _subagent_run("run_a")}},
            tool_call_id="wait_1",
        )
    )
    content = command.update["subagent_runs"]["run_a"]["report"]["content"]

    assert core._count_text_tokens(content) <= 8
    assert "[truncated to approximately 8 tokens]" in content


def test_await_wait_stores_tool_calls_from_error_run() -> None:
    client = _AsyncCompletedRunsClient(status="error")
    tool = _build_wait_tool(_ClientCacheStub(async_client=client))
    runtime = SimpleNamespace(
        state={"subagent_runs": {"run_a": _subagent_run("run_a")}},
        tool_call_id="wait_1",
    )

    assert tool.coroutine is not None
    command = asyncio.run(tool.coroutine(runtime=runtime))

    subagent_run = command.update["subagent_runs"]["run_a"]
    assert subagent_run["tool_calls"] == [
        {"id": "run_a_call", "name": "resource_tool", "args": {"run": "run_a"}},
    ]
    assert subagent_run["report"]["content"] == "subagent failed"


def test_await_wait_polls_until_a_run_completes(monkeypatch) -> None:
    monkeypatch.setattr(core, "WAIT_POLL_SECONDS", 0.0)
    client = SimpleNamespace(
        runs=SimpleNamespace(
            get=AsyncMock(
                side_effect=[
                    {
                        "thread_id": "run_a_thread",
                        "run_id": "run_a",
                        "status": "running",
                    },
                    {
                        "thread_id": "run_a_thread",
                        "run_id": "run_a",
                        "status": "success",
                    },
                ]
            )
        ),
        threads=SimpleNamespace(
            get_history=AsyncMock(
                return_value=_CompletedThreads().get_history(
                    thread_id="run_a_thread",
                    limit=core.HISTORY_LIMIT,
                    metadata={"run_id": "run_a"},
                )
            )
        ),
    )
    tool = _build_wait_tool(_ClientCacheStub(async_client=client))
    runtime = SimpleNamespace(
        state={"subagent_runs": {"run_a": _subagent_run("run_a")}},
        tool_call_id="wait_1",
    )

    assert tool.coroutine is not None
    command = asyncio.run(tool.coroutine(runtime=runtime))

    assert client.runs.get.await_count == 2
    assert command.update["subagent_runs"]["run_a"]["report"]["content"] == (
        "report from run_a"
    )


@pytest.mark.parametrize("error_type", [ReadError, RemoteProtocolError])
def test_await_wait_retries_transient_run_errors(
    monkeypatch, error_type: type[Exception]
) -> None:
    monkeypatch.setattr(core, "WAIT_POLL_SECONDS", 0.0)
    client = SimpleNamespace(
        runs=SimpleNamespace(
            get=AsyncMock(
                side_effect=[
                    error_type("transient error", request=Request("GET", "http://test")),
                    {
                        "thread_id": "run_a_thread",
                        "run_id": "run_a",
                        "status": "success",
                    },
                ]
            )
        ),
        threads=SimpleNamespace(
            get_history=AsyncMock(
                return_value=_CompletedThreads().get_history(
                    thread_id="run_a_thread",
                    limit=core.HISTORY_LIMIT,
                    metadata={"run_id": "run_a"},
                )
            )
        ),
    )
    tool = _build_wait_tool(_ClientCacheStub(async_client=client))
    runtime = SimpleNamespace(
        state={"subagent_runs": {"run_a": _subagent_run("run_a")}},
        tool_call_id="wait_1",
    )

    assert tool.coroutine is not None
    command = asyncio.run(tool.coroutine(runtime=runtime))

    assert client.runs.get.await_count == 2
    assert command.update["subagent_runs"]["run_a"]["report"]["content"] == (
        "report from run_a"
    )


def test_await_wait_raises_unexpected_run_errors() -> None:
    error = RuntimeError("unexpected error")
    client = SimpleNamespace(
        runs=SimpleNamespace(get=AsyncMock(side_effect=error)),
    )
    tool = _build_wait_tool(_ClientCacheStub(async_client=client))
    runtime = SimpleNamespace(
        state={"subagent_runs": {"run_a": _subagent_run("run_a")}},
        tool_call_id="wait_1",
    )

    assert tool.coroutine is not None
    with pytest.raises(RuntimeError, match="unexpected error"):
        asyncio.run(tool.coroutine(runtime=runtime))


@pytest.mark.parametrize(
    ("tool_names", "rejected_ids"),
    [
        (["wait", "spawn_subagent"], ["call_0"]),
        (["wait", "wait"], ["call_0", "call_1"]),
    ],
)
def test_after_model_rejects_parallel_wait_calls(
    tool_names: list[str], rejected_ids: list[str]
) -> None:
    middleware = DecomposerAgentMiddleware([SUBAGENT_TYPE], 1)
    tool_calls = [
        {
            "name": name,
            "args": {},
            "id": f"call_{index}",
            "type": "tool_call",
        }
        for index, name in enumerate(tool_names)
    ]

    update = middleware.after_model(
        {"messages": [AIMessage(content="", tool_calls=tool_calls)]},
        SimpleNamespace(),
    )

    assert update is not None
    assert [message.tool_call_id for message in update["messages"]] == rejected_ids
    assert all(isinstance(message, ToolMessage) for message in update["messages"])
    assert all(message.name == "wait" for message in update["messages"])
    assert all(
        message.content == PARALLEL_WAIT_CALL_ERROR
        for message in update["messages"]
    )


def test_after_model_allows_non_wait_tool_calls() -> None:
    middleware = DecomposerAgentMiddleware([SUBAGENT_TYPE], 1)
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "spawn_subagent",
                "args": {},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )

    assert (
        middleware.after_model(
            {"messages": [ai_message]},
            SimpleNamespace(),
        )
        is None
    )


def test_after_model_rejects_report_before_all_subagents_finish() -> None:
    middleware = DecomposerAgentMiddleware([SUBAGENT_TYPE], 1)

    update = middleware.after_model(
        {
            "messages": [AIMessage(content="early report")],
            "subagent_runs": {"run_a": _subagent_run("run_a")},
        },
        SimpleNamespace(),
    )

    assert update is not None
    assert update["jump_to"] == "model"
    assert update["messages"] == [HumanMessage(content=EARLY_REPORT_ERROR)]


def test_after_model_rejects_empty_report() -> None:
    middleware = DecomposerAgentMiddleware([SUBAGENT_TYPE], 1)

    update = middleware.after_model(
        {"messages": [AIMessage(content=" \n ")]},
        SimpleNamespace(),
    )

    assert update is not None
    assert update["jump_to"] == "model"
    assert update["messages"] == [HumanMessage(content=EMPTY_REPORT_ERROR)]


def test_after_model_accepts_report_after_all_subagents_finish() -> None:
    middleware = DecomposerAgentMiddleware([SUBAGENT_TYPE], 1)

    assert (
        middleware.after_model(
            {
                "messages": [AIMessage(content="final report")],
                "subagent_runs": {
                    "run_a": _completed_subagent_run("run_a"),
                },
            },
            SimpleNamespace(),
        )
        is None
    )


def test_after_model_ignores_invalid_tool_calls_when_checking_report() -> None:
    middleware = DecomposerAgentMiddleware([SUBAGENT_TYPE], 1)
    ai_message = AIMessage(
        content="early report",
        invalid_tool_calls=[
            {
                "name": "wait",
                "args": "{",
                "id": "invalid_call",
                "error": "invalid JSON",
                "type": "invalid_tool_call",
            }
        ],
    )

    assert (
        middleware.after_model(
            {"messages": [ai_message]},
            SimpleNamespace(),
        )
        is None
    )

    update = middleware.after_model(
        {
            "messages": [ai_message],
            "subagent_runs": {"run_a": _subagent_run("run_a")},
        },
        SimpleNamespace(),
    )

    assert update is not None
    assert update["messages"] == [HumanMessage(content=EARLY_REPORT_ERROR)]


def test_after_model_uses_latest_ai_message() -> None:
    middleware = DecomposerAgentMiddleware([SUBAGENT_TYPE], 1)

    update = middleware.after_model(
        {
            "messages": [
                AIMessage(content="old report"),
                HumanMessage(content="continue"),
                AIMessage(content=""),
            ]
        },
        SimpleNamespace(),
    )

    assert update is not None
    assert update["messages"] == [HumanMessage(content=EMPTY_REPORT_ERROR)]


def _subagent_run(run_id: str) -> dict[str, Any]:
    return {
        "subagent_run_id": run_id,
        "subagent_type_id": "test",
        "assistant_id": "test_assistant",
        "thread_id": f"{run_id}_thread",
        "run_id": run_id,
        "status": "running",
        "prompt": "do the task",
    }


def _completed_subagent_run(run_id: str) -> dict[str, Any]:
    return {
        **_subagent_run(run_id),
        "status": "success",
        "report": {
            "subagent_run_id": run_id,
            "status": "success",
            "content": "done",
        },
    }


class _ClientCacheStub:
    def __init__(self, *, sync_client: Any = None, async_client: Any = None) -> None:
        self.sync_client = sync_client
        self.async_client = async_client

    def get_sync(self, subagent_type_id: str) -> Any:
        assert subagent_type_id == "test"
        return self.sync_client

    def get_async(self, subagent_type_id: str) -> Any:
        assert subagent_type_id == "test"
        return self.async_client


class _SyncThreads:
    def create(self) -> dict[str, str]:
        return {"thread_id": "thread_1"}


class _SyncRuns:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> dict[str, str]:
        self.create_kwargs = kwargs
        return {"run_id": "run_1", "status": "pending"}


class _SyncClient:
    def __init__(self) -> None:
        self.threads = _SyncThreads()
        self.runs = _SyncRuns()


class _AsyncThreads:
    async def create(self) -> dict[str, str]:
        return {"thread_id": "thread_1"}


class _AsyncRuns:
    def __init__(self) -> None:
        self.create_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> dict[str, str]:
        self.create_kwargs = kwargs
        return {"run_id": "run_1", "status": "pending"}


class _AsyncClient:
    def __init__(self) -> None:
        self.threads = _AsyncThreads()
        self.runs = _AsyncRuns()


class _CompletedRuns:
    def __init__(self, status: str = "success") -> None:
        self.status = status

    def get(self, *, thread_id: str, run_id: str) -> dict[str, str]:
        assert thread_id == f"{run_id}_thread"
        run = {
            "thread_id": thread_id,
            "run_id": run_id,
            "status": self.status,
        }
        if self.status == "error":
            run["error"] = "subagent failed"
        return run


class _CompletedThreads:
    def get_history(self, *, thread_id: str, limit: int, metadata: dict[str, str]) -> list[dict[str, Any]]:
        run_id = metadata["run_id"]
        assert thread_id == f"{run_id}_thread"
        input_message = {"type": "human", "content": "do the task"}
        return [
            {
                "metadata": {"source": "loop"},
                "values": {
                    "messages": [
                        input_message,
                        {
                            "type": "ai",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": f"{run_id}_call",
                                    "name": "resource_tool",
                                    "args": {"run": run_id},
                                }
                            ],
                        },
                        {"type": "tool", "content": "result"},
                        {"type": "ai", "content": f"report from {run_id}", "tool_calls": []},
                    ]
                },
            },
            {
                "metadata": {"source": "input"},
                "values": {"messages": [input_message]},
            },
        ]


class _CompletedRunsClient:
    def __init__(self, status: str = "success") -> None:
        self.runs = _CompletedRuns(status)
        self.threads = _CompletedThreads()


class _AsyncCompletedRuns:
    def __init__(self, status: str = "success") -> None:
        self.status = status

    async def get(self, *, thread_id: str, run_id: str) -> dict[str, str]:
        return _CompletedRuns(self.status).get(thread_id=thread_id, run_id=run_id)


class _AsyncCompletedThreads:
    async def get_history(
        self,
        *,
        thread_id: str,
        limit: int,
        metadata: dict[str, str],
    ) -> list[dict[str, Any]]:
        return _CompletedThreads().get_history(
            thread_id=thread_id,
            limit=limit,
            metadata=metadata,
        )


class _AsyncCompletedRunsClient:
    def __init__(self, status: str = "success") -> None:
        self.runs = _AsyncCompletedRuns(status)
        self.threads = _AsyncCompletedThreads()
