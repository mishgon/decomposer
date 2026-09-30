import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import decomposer.core as core
import pytest
from httpx import ReadError, RemoteProtocolError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from decomposer import create_decomposer_agent
from decomposer.core import (
    DecomposerAgentMiddleware,
    SubagentType,
    _build_new_tool,
    _build_fork_tool,
    _build_run_tool,
    _build_wait_tool,
)
from decomposer.prompts import (
    PARALLEL_RUN_CALL_ERROR,
    PARALLEL_FORK_RUN_CALL_ERROR,
    EARLY_RESPONSE_ERROR,
    EMPTY_RESPONSE_ERROR,
    NO_ACTIVE_RUNS_ERROR,
    PARALLEL_WAIT_CALL_ERROR,
)


SUBAGENT_TYPE: SubagentType = {
    "subagent_type_id": "test",
    "description": "Test subagent",
    "assistant_id": "test_assistant",
    "url": "http://subagents.test",
}


SUBAGENT_TYPES = [
    {
        "subagent_type_id": "dummy",
        "description": "Dummy subagent for smoke testing.",
        "assistant_id": "dummy",
        "url": "http://unused",
    }
]


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_new_rejects_unknown_type() -> None:
    client = MagicMock()
    tool = _build_new_tool(
        {"test": SUBAGENT_TYPE},
        _client_cache(client),
    )
    runtime = SimpleNamespace(tool_call_id="create_1")

    result = tool.func("unknown", runtime)

    assert "Unknown subagent type ID `unknown`" in result
    client.threads.create.assert_not_called()


@pytest.mark.parametrize("async_invocation", [False, True])
def test_fork_copies_thread_and_preserves_subagent_type(async_invocation: bool) -> None:
    client = AsyncMock() if async_invocation else MagicMock()
    client.threads.copy.return_value = {"thread_id": "copied_thread"}
    source = _subagent("source_thread")
    runtime = SimpleNamespace(
        state={"subagents": {"source_thread": source}, "subagent_runs": {}},
        tool_call_id="fork_1",
    )
    tool = _build_fork_tool(_client_cache(client))

    if async_invocation:
        command = asyncio.run(tool.coroutine("source_thread", runtime))
    else:
        command = tool.func("source_thread", runtime)

    client.threads.copy.assert_called_once_with("source_thread")
    assert command.update["subagents"] == {
        "copied_thread": {
            **source,
            "subagent_id": "copied_thread",
            "thread_id": "copied_thread",
        }
    }
    assert source == _subagent("source_thread")
    message = command.update["messages"][0]
    assert message.tool_call_id == "fork_1"
    assert json.loads(message.content) == {"subagent_id": "copied_thread"}


@pytest.mark.parametrize("async_invocation", [False, True])
def test_run_records_run_and_passes_context(async_invocation: bool) -> None:
    client = AsyncMock() if async_invocation else MagicMock()
    client.runs.create.return_value = {"run_id": "run_1", "status": "pending"}
    tool = _build_run_tool(_client_cache(client), 1)
    context = {"value": 42}
    runtime = SimpleNamespace(
        state={"subagents": {"thread_1": _subagent("thread_1")}, "subagent_runs": {}},
        context=context,
        tool_call_id="prompt_1",
    )

    if async_invocation:
        command = asyncio.run(tool.coroutine("thread_1", "do the task", runtime))
    else:
        command = tool.func("thread_1", "do the task", runtime)

    client.runs.create.assert_called_once_with(
        thread_id="thread_1",
        assistant_id="test_assistant",
        input={"messages": [{"role": "user", "content": "do the task"}]},
        config={"recursion_limit": 1},
        context=context,
        multitask_strategy="reject",
    )
    assert "subagents" not in command.update
    assert command.update["subagent_runs"] == {
        "run_1": {
            "subagent_run_id": "run_1",
            "subagent_id": "thread_1",
            "run_id": "run_1",
            "status": "pending",
            "prompt": "do the task",
        }
    }
    assert json.loads(command.update["messages"][0].content) == {
        "subagent_run_id": "run_1",
    }


@pytest.mark.parametrize(
    ("status", "collected", "expected_error"),
    [
        (None, False, "Unknown subagent ID `run_1_thread`"),
        ("running", False, "active run `run_1`"),
        ("error", True, "status `\"error\"`"),
    ],
)
@pytest.mark.parametrize("tool_name", ["run", "fork"])
def test_subagent_tools_reject_unavailable_subagent(
    status: str | None, collected: bool, expected_error: str, tool_name: str
) -> None:
    client = MagicMock()
    tool = (
        _build_run_tool(_client_cache(client), 1)
        if tool_name == "run"
        else _build_fork_tool(_client_cache(client))
    )
    state = {"subagents": {}, "subagent_runs": {}}
    if status is not None:
        state["subagents"]["run_1_thread"] = _subagent("run_1_thread")
        run = _completed_subagent_run("run_1") if collected else _subagent_run("run_1")
        run["status"] = status
        state["subagent_runs"]["run_1"] = run
    runtime = SimpleNamespace(state=state, context=None, tool_call_id="prompt_1")

    args = {"subagent_id": "run_1_thread", "runtime": runtime}
    if tool_name == "run":
        args["prompt"] = "follow up"
    result = tool.func(**args)

    assert expected_error in result
    client.runs.create.assert_not_called()
    client.threads.create.assert_not_called()
    client.threads.copy.assert_not_called()


def test_wait_stores_tool_calls_in_returned_response_order() -> None:
    client = _completed_client()
    tool = _build_wait_tool(_client_cache(client))
    runtime = SimpleNamespace(
        state={
            "subagents": {
                f"{run_id}_thread": _subagent(f"{run_id}_thread")
                for run_id in ("run_c", "run_b", "run_a")
            },
            "subagent_runs": {
                "run_c": {
                    **_subagent_run("run_c"),
                    "status": "responded",
                    "response": "earlier response",
                    "response_sequence_number": 0,
                },
                "run_b": _subagent_run("run_b"),
                "run_a": _subagent_run("run_a"),
            }
        },
        tool_call_id="wait_1",
    )

    command = tool.func(runtime=runtime)

    assert command.update["subagent_runs"]["run_b"]["response_sequence_number"] == 1
    assert command.update["subagent_runs"]["run_a"]["response_sequence_number"] == 2
    assert command.update["subagent_runs"]["run_b"]["tool_calls"] == [
        {"id": "run_b_call", "name": "resource_tool", "args": {"run": "run_b"}},
    ]
    assert command.update["subagent_runs"]["run_a"]["tool_calls"] == [
        {"id": "run_a_call", "name": "resource_tool", "args": {"run": "run_a"}},
    ]
    assert json.loads(command.update["messages"][0].content) == [
        {
            "subagent_id": f"{run_id}_thread",
            "subagent_run_id": run_id,
            "status": "responded",
            "response": f"response from {run_id}",
            "error": None,
        }
        for run_id in ("run_b", "run_a")
    ]


@pytest.mark.parametrize(
    "last_message",
    [
        {"type": "tool", "content": "result"},
        {"type": "ai", "content": "", "tool_calls": [
            {"id": "unfinished_call", "name": "resource_tool", "args": {}},
        ]},
        {"type": "ai", "content": None, "tool_calls": []},
    ],
    ids=["tool-message", "pending-tool-call", "null-content"],
)
def test_wait_rejects_responded_run_without_final_response(last_message) -> None:
    client = _completed_client()
    history = _run_history("run_a")
    history[0]["values"]["messages"][-1] = last_message
    client.threads.get_history.side_effect = None
    client.threads.get_history.return_value = history
    tool = _build_wait_tool(_client_cache(client))
    runtime = _wait_runtime()

    with pytest.raises(ValueError, match="Expected a final AI message.*responded run `run_a`"):
        tool.func(runtime)


def test_wait_accepts_empty_final_response() -> None:
    client = _completed_client()
    history = _run_history("run_a")
    history[0]["values"]["messages"][-1]["content"] = ""
    client.threads.get_history.side_effect = None
    client.threads.get_history.return_value = history
    tool = _build_wait_tool(_client_cache(client))
    runtime = _wait_runtime()

    command = tool.func(runtime)

    run = command.update["subagent_runs"]["run_a"]
    assert run["status"] == "responded"
    assert run["response"] == ""
    assert json.loads(command.update["messages"][0].content)[0]["response"] == ""


def test_wait_stores_tool_calls_from_error_run() -> None:
    client = _completed_client(status="error")
    tool = _build_wait_tool(_client_cache(client))
    runtime = _wait_runtime()

    command = tool.func(runtime)

    subagent_run = command.update["subagent_runs"]["run_a"]
    assert subagent_run["tool_calls"] == [
        {"id": "run_a_call", "name": "resource_tool", "args": {"run": "run_a"}},
    ]
    assert subagent_run["response"] is None
    assert json.loads(subagent_run["error"]) == {
        "error": "RuntimeError", "message": "subagent failed",
    }


@pytest.mark.parametrize(
    ("empty_history", "thread", "expected_error"),
    [
        (False, {}, None),
        (
            True,
            {"error": {"error": "ValueError", "message": 'ошибка "value"\nnext line'}},
            r'{"error": "ValueError", "message": "ошибка \"value\"\nnext line"}',
        ),
    ],
)
def test_wait_formats_thread_error(empty_history, thread, expected_error) -> None:
    history = [] if empty_history else _run_history("run_a")
    client = MagicMock()
    client.runs.get.return_value = {
        "thread_id": "run_a_thread", "run_id": "run_a", "status": "error",
    }
    client.threads.get_history.return_value = history
    client.threads.get.return_value = thread
    tool = _build_wait_tool(_client_cache(client))
    runtime = _wait_runtime()

    command = tool.func(runtime=runtime)

    run = command.update["subagent_runs"]["run_a"]
    assert run["status"] == "error"
    assert run["response"] is None
    assert run["error"] == expected_error
    assert run["response_sequence_number"] == 0
    assert json.loads(command.update["messages"][0].content) == [{
        "subagent_id": "run_a_thread", "subagent_run_id": "run_a",
        "status": "error", "response": None, "error": expected_error,
    }]
    runtime.state["subagent_runs"].update(command.update["subagent_runs"])
    assert core._get_current_subagent_runs(runtime.state["subagent_runs"]) == {}
    assert "status `\"error\"`" in core._get_subagent_error("run_a_thread", runtime.state)
    middleware = DecomposerAgentMiddleware([SUBAGENT_TYPE], 1)
    assert middleware.after_model(
        {**runtime.state, "messages": [AIMessage(content="Failed.")]}, SimpleNamespace()
    ) is None
    client.threads.get.assert_called_once_with(thread_id="run_a_thread")
    client.threads.get_history.assert_called_once_with(
        thread_id="run_a_thread", limit=core.HISTORY_LIMIT, metadata={"run_id": "run_a"},
    )


@pytest.mark.parametrize("first_result", [
    {"status": "running"},
    ReadError("transient error"),
    RemoteProtocolError("transient error"),
], ids=["running", "read-error", "protocol-error"])
def test_await_wait_retries_until_completion(monkeypatch, first_result) -> None:
    monkeypatch.setattr(core, "WAIT_POLL_SECONDS", 0.0)
    client = _completed_client(async_invocation=True)
    client.runs.get.side_effect = [first_result, {
        "thread_id": "run_a_thread", "run_id": "run_a", "status": "success",
    }]
    tool = _build_wait_tool(_client_cache(client))
    runtime = _wait_runtime()

    command = asyncio.run(tool.coroutine(runtime))

    assert client.runs.get.await_count == 2
    assert command.update["subagent_runs"]["run_a"]["response"] == "response from run_a"


def test_decomposer_waits_before_creating_subagents() -> None:
    agent = create_decomposer_agent(
        decomposer_model=ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[_call("wait", "wait-call")],
                ),
                AIMessage(content="done"),
            ]
        ),
        subagent_types=SUBAGENT_TYPES,
    )
    inputs = {"messages": [{"role": "user", "content": "Hello"}]}

    result = agent.invoke(inputs)

    assert result["subagents"] == {}
    assert result["subagent_runs"] == {}
    assert any(
        isinstance(message, ToolMessage)
        and message.tool_call_id == "wait-call"
        and message.content == NO_ACTIVE_RUNS_ERROR
        for message in result["messages"]
    )
    assert result["messages"][-1].content == "done"


@pytest.mark.parametrize("async_invocation", [False, True])
def test_decomposer_delegates(monkeypatch, async_invocation: bool) -> None:
    client = _mock_client(monkeypatch, async_invocation)
    client.threads.create.side_effect = [
        {"thread_id": f"thread_{i}"} for i in range(2)
    ]
    client.runs.create.side_effect = lambda thread_id, **kwargs: {
        "run_id": f"{thread_id}_run",
        "status": "pending",
    }
    client.runs.get.side_effect = lambda thread_id, run_id: {
        "thread_id": thread_id,
        "run_id": run_id,
        "status": "success",
    }
    input_message = {"type": "human", "content": "Say hello."}
    client.threads.get_history.return_value = [
        {
            "metadata": {"source": "loop"},
            "values": {
                "messages": [
                    input_message,
                    {"type": "ai", "content": "hello", "tool_calls": []},
                ]
            },
        },
        {
            "metadata": {"source": "input"},
            "values": {"messages": [input_message]},
        },
    ]
    agent = create_decomposer_agent(
        decomposer_model=ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        _call("new", f"create-call-{i}", subagent_type_id="dummy")
                        for i in range(2)
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        _call(
                            "run", f"prompt-call-{i}",
                            subagent_id=f"thread_{i}", prompt="Say hello.",
                        )
                        for i in range(2)
                    ],
                ),
                AIMessage(content="early response"),
                AIMessage(
                    content="",
                    tool_calls=[_call("wait", "wait-call")],
                ),
                AIMessage(content="dummy"),
            ]
        ),
        subagent_types=SUBAGENT_TYPES,
    )

    inputs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = asyncio.run(agent.ainvoke(inputs)) if async_invocation else agent.invoke(inputs)

    assert client.runs.create.call_count == 2
    assert all(
        call.kwargs["config"] is None
        for call in client.runs.create.call_args_list
    )
    assert result["subagents"] == {
        f"thread_{i}": {
            "subagent_id": f"thread_{i}",
            "subagent_type_id": "dummy",
            "assistant_id": "dummy",
            "thread_id": f"thread_{i}",
        }
        for i in range(2)
    }
    assert len(result["subagent_runs"]) == 2
    for i in range(2):
        run = result["subagent_runs"][f"thread_{i}_run"]
        assert run["subagent_id"] == f"thread_{i}"
        assert run["prompt"] == "Say hello."
        assert run["status"] == "responded"
        assert run["response"] == "hello"
        assert run["error"] is None
    responses = next(
        json.loads(message.content)
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "wait-call"
    )
    assert {response["subagent_id"] for response in responses} == set(result["subagents"])
    assert any(
        isinstance(message, HumanMessage) and message.content == EARLY_RESPONSE_ERROR
        for message in result["messages"]
    )
    assert result["messages"][-1].content == "dummy"


def test_decomposer_rejects_parallel_wait_and_finishes_with_idle_subagent(monkeypatch) -> None:
    client = _mock_client(monkeypatch, False)
    client.threads.create.return_value = {"thread_id": "dummy-thread"}
    agent = create_decomposer_agent(
        decomposer_model=ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        _call("wait", "wait-call"),
                        _call("new", "create-call", subagent_type_id="dummy"),
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        subagent_types=SUBAGENT_TYPES,
    )
    inputs = {"messages": [{"role": "user", "content": "Hello"}]}

    result = agent.invoke(inputs)

    client.threads.create.assert_called_once_with()
    client.runs.create.assert_not_called()
    client.runs.get.assert_not_called()
    assert list(result["subagents"]) == ["dummy-thread"]
    assert result["subagent_runs"] == {}
    rejected_calls = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.content == PARALLEL_WAIT_CALL_ERROR
    ]
    assert [message.tool_call_id for message in rejected_calls] == ["wait-call"]
    assert result["messages"][-1].content == "done"


def test_decomposer_rejects_duplicate_runs_and_parallel_waits(monkeypatch) -> None:
    client = _mock_client(monkeypatch, False)
    client.threads.create.return_value = {"thread_id": "thread_1"}
    agent = create_decomposer_agent(
        decomposer_model=ToolCallingFakeModel(responses=[
            AIMessage(content="", tool_calls=[_call("new", "create", subagent_type_id="dummy")]),
            AIMessage(content="", tool_calls=[
                _call("wait", "wait_0"),
                _call("wait", "wait_1"),
                *[
                    _call("run", f"prompt_{i}", subagent_id="thread_1", prompt=f"Task {i}")
                    for i in range(2)
                ],
            ]),
            AIMessage(content="done"),
        ]),
        subagent_types=SUBAGENT_TYPES,
    )
    inputs = {"messages": [{"role": "user", "content": "Hello"}]}

    result = agent.invoke(inputs)

    client.runs.create.assert_not_called()
    client.runs.get.assert_not_called()
    assert result["subagent_runs"] == {}
    rejections = {
        message.tool_call_id: message.content
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id != "create"
    }
    assert rejections == {
        "wait_0": PARALLEL_WAIT_CALL_ERROR,
        "wait_1": PARALLEL_WAIT_CALL_ERROR,
        "prompt_0": PARALLEL_RUN_CALL_ERROR,
        "prompt_1": PARALLEL_RUN_CALL_ERROR,
    }
    assert result["messages"][-1].content == "done"


def test_decomposer_retries_empty_response_asynchronously() -> None:
    agent = create_decomposer_agent(
        decomposer_model=ToolCallingFakeModel(
            responses=[
                AIMessage(content=" \n "),
                AIMessage(content="done"),
            ]
        ),
        subagent_types=SUBAGENT_TYPES,
    )

    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})
    )

    assert any(
        isinstance(message, HumanMessage) and message.content == EMPTY_RESPONSE_ERROR
        for message in result["messages"]
    )
    assert result["messages"][-1].content == "done"


def _subagent(subagent_id: str) -> dict[str, Any]:
    return {
        "subagent_id": subagent_id,
        "subagent_type_id": "test",
        "assistant_id": "test_assistant",
        "thread_id": subagent_id,
    }


def _subagent_run(run_id: str) -> dict[str, Any]:
    return {
        "subagent_run_id": run_id,
        "subagent_id": f"{run_id}_thread",
        "run_id": run_id,
        "status": "running",
        "prompt": "do the task",
    }


def _completed_subagent_run(run_id: str) -> dict[str, Any]:
    return {
        **_subagent_run(run_id),
        "status": "responded",
        "response": "done",
        "response_sequence_number": 0,
    }


def _client_cache(client):
    return SimpleNamespace(get_sync=lambda _: client, get_async=lambda _: client)


def _completed_client(async_invocation=False, status="success"):
    client = AsyncMock() if async_invocation else MagicMock()
    client.runs.get.side_effect = lambda thread_id, run_id: {
        "thread_id": thread_id, "run_id": run_id, "status": status,
    }
    client.threads.get.return_value = {
        "error": {"error": "RuntimeError", "message": "subagent failed"},
    }
    client.threads.get_history.side_effect = (
        lambda thread_id, limit, metadata: _run_history(metadata["run_id"])
    )
    return client


def _run_history(run_id):
    input_message = {"type": "human", "content": "do the task"}
    return [
        {
            "metadata": {"source": "loop"},
            "values": {"messages": [
                input_message,
                {"type": "ai", "content": "", "tool_calls": [
                    {"id": f"{run_id}_call", "name": "resource_tool", "args": {"run": run_id}},
                ]},
                {"type": "tool", "content": "result"},
                {"type": "ai", "content": f"response from {run_id}", "tool_calls": []},
            ]},
        },
        {"metadata": {"source": "input"}, "values": {"messages": [input_message]}},
    ]


def _wait_runtime():
    return SimpleNamespace(
        state={
            "subagents": {"run_a_thread": _subagent("run_a_thread")},
            "subagent_runs": {"run_a": _subagent_run("run_a")},
        },
        tool_call_id="wait_1",
    )


def _mock_client(monkeypatch, async_invocation):
    client = AsyncMock() if async_invocation else MagicMock()
    monkeypatch.setattr(
        "decomposer.core.get_client" if async_invocation else "decomposer.core.get_sync_client",
        lambda **kwargs: client,
    )
    return client


def _call(name, call_id, **args):
    return {"name": name, "id": call_id, "args": args}


def test_decomposer_rejects_parallel_fork_and_run(monkeypatch) -> None:
    client = _mock_client(monkeypatch, False)
    agent = create_decomposer_agent(
        decomposer_model=ToolCallingFakeModel(responses=[
            AIMessage(content="", tool_calls=[
                _call("run", "run", subagent_id="thread_1", prompt="Task"),
                _call("fork", "fork", subagent_id="thread_1"),
            ]),
            AIMessage(content="done"),
        ]),
        subagent_types=SUBAGENT_TYPES,
    )
    inputs = {
        "messages": [HumanMessage("Hello")],
        "subagents": {"thread_1": _subagent("thread_1")},
    }
    result = agent.invoke(inputs)
    client.runs.create.assert_not_called()
    client.threads.copy.assert_not_called()
    assert [m.content for m in result["messages"] if isinstance(m, ToolMessage)] == [
        PARALLEL_FORK_RUN_CALL_ERROR, PARALLEL_FORK_RUN_CALL_ERROR,
    ]
