import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import decomposer.core as core
import pytest
from httpx import ReadError, RemoteProtocolError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from decomposer import create_decomposer_agent
from decomposer.core import (
    DecomposerAgentMiddleware,
    AgentType,
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
    WAIT_TIMEOUT_ERROR,
)


AGENT_TYPE: AgentType = {
    "agent_type_id": "test",
    "description": "Test agent",
    "assistant_id": "test_assistant",
    "url": "http://agents.test",
}


AGENT_TYPES = [
    {
        "agent_type_id": "dummy",
        "description": "Dummy agent for smoke testing.",
        "assistant_id": "dummy",
        "url": "http://unused",
    }
]


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("async_invocation", [False, True])
def test_decomposer_runs_preserve_invocations_in_one_thread(monkeypatch, async_invocation):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(core.time, "time", lambda: clock.now)

    class TimedModel(ToolCallingFakeModel):
        def _generate(self, *args, **kwargs):
            clock.now += 100.0
            return super()._generate(*args, **kwargs)

    agent = create_decomposer_agent(
        decomposer_model=TimedModel(responses=[
            AIMessage(content="First answer."),
            AIMessage(content=[{"type": "text", "text": "Second answer."}]),
            AIMessage(content="Other thread."),
        ]),
        agent_types=AGENT_TYPES,
        checkpointer=InMemorySaver(),
    )

    def invoke(messages, config):
        inputs = {"messages": messages}
        return asyncio.run(agent.ainvoke(inputs, config)) if async_invocation else agent.invoke(inputs, config)

    config = {"configurable": {"thread_id": "conversation"}}
    first_id = UUID("00000000-0000-0000-0000-000000000001")
    first = invoke(
        [HumanMessage(content="Old request."), AIMessage(content="Old answer."),
         HumanMessage(content="First request.")],
        {**config, "run_id": first_id},
    )
    expected_first = {
        "agent_run_id": str(first_id), "agent_id": "conversation", "run_id": str(first_id),
        "status": "responded", "prompt": "First request.", "response": "First answer.",
        "started_at": 100.0, "collected_at": 200.0,
    }
    assert first["decomposer_agent_runs"] == [expected_first]

    clock.now = 300.0
    second = invoke([HumanMessage(content=[{"type": "text", "text": "Second request."}])], config)
    runs = second["decomposer_agent_runs"]
    assert len(runs) == 2
    assert runs[0] == expected_first
    assert first["decomposer_agent_runs"] == [expected_first]
    second_id = str(UUID(runs[1]["agent_run_id"]))
    assert second_id != str(first_id)
    assert runs[1] == {
        "agent_run_id": second_id, "agent_id": "conversation", "run_id": second_id,
        "status": "responded", "prompt": "Second request.", "response": "Second answer.",
        "started_at": 300.0, "collected_at": 400.0,
    }
    pending = next(
        snapshot.values["decomposer_agent_runs"]
        for snapshot in agent.get_state_history(config)
        if len(snapshot.values.get("decomposer_agent_runs", [])) == 2
        and snapshot.values["decomposer_agent_runs"][-1]["status"] == "running"
    )
    assert pending[0] == expected_first
    assert pending[1]["agent_run_id"] == second_id
    assert "collected_at" not in pending[1]
    assert "response" not in pending[1]

    other = invoke([HumanMessage(content="Other request.")], {"configurable": {"thread_id": "other"}})
    assert len(other["decomposer_agent_runs"]) == 1
    assert other["decomposer_agent_runs"][0]["agent_id"] == "other"
    assert agent.get_state(config).values["decomposer_agent_runs"] == runs


def test_new_rejects_unknown_type() -> None:
    client = MagicMock()
    tool = _build_new_tool(
        {"test": AGENT_TYPE},
        _client_cache(client),
    )
    runtime = SimpleNamespace(tool_call_id="create_1")

    result = tool.func("unknown", runtime)

    assert "Unknown agent type ID `unknown`" in result
    client.threads.create.assert_not_called()


@pytest.mark.parametrize("async_invocation", [False, True])
def test_new_records_creation_time(monkeypatch, async_invocation: bool) -> None:
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(core.time, "time", lambda: clock.now)
    client = AsyncMock() if async_invocation else MagicMock()

    def create_thread():
        clock.now = 200.0
        return {"thread_id": "thread_1"}

    client.threads.create.side_effect = create_thread
    tool = _build_new_tool({"test": AGENT_TYPE}, _client_cache(client))
    runtime = SimpleNamespace(tool_call_id="create_1")

    if async_invocation:
        command = asyncio.run(tool.coroutine("test", runtime))
    else:
        command = tool.func("test", runtime)

    assert command.update["agents"] == {
        "thread_1": {**_agent("thread_1"), "created_at": 200.0},
    }


@pytest.mark.parametrize("async_invocation", [False, True])
@pytest.mark.parametrize("forked_from", [None, "original_thread"])
def test_fork_copies_thread_and_preserves_agent_type(
    monkeypatch, async_invocation: bool, forked_from: str | None,
) -> None:
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(core.time, "time", lambda: clock.now)
    client = AsyncMock() if async_invocation else MagicMock()

    def copy_thread(thread_id):
        clock.now = 200.0
        return {"thread_id": "copied_thread"}

    client.threads.copy.side_effect = copy_thread
    source = _agent("source_thread")
    if forked_from is not None:
        source["forked_from"] = forked_from
    original_source = source.copy()
    runtime = SimpleNamespace(
        state={"agents": {"source_thread": source}, "agent_runs": {}},
        tool_call_id="fork_1",
    )
    tool = _build_fork_tool(_client_cache(client))

    if async_invocation:
        command = asyncio.run(tool.coroutine("source_thread", runtime))
    else:
        command = tool.func("source_thread", runtime)

    client.threads.copy.assert_called_once_with("source_thread")
    assert command.update["agents"] == {
        "copied_thread": {
            **source,
            "agent_id": "copied_thread",
            "thread_id": "copied_thread",
            "created_at": 200.0,
            "forked_from": "source_thread",
        }
    }
    assert source == original_source
    message = command.update["messages"][0]
    assert message.tool_call_id == "fork_1"
    assert json.loads(message.content) == {"agent_id": "copied_thread"}


@pytest.mark.parametrize("async_invocation", [False, True])
def test_run_records_run_and_passes_context(monkeypatch, async_invocation: bool) -> None:
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(core.time, "time", lambda: clock.now)
    client = AsyncMock() if async_invocation else MagicMock()

    def create_run(**kwargs):
        clock.now = 200.0
        return {"run_id": "run_1", "status": "pending"}

    client.runs.create.side_effect = create_run
    tool = _build_run_tool(_client_cache(client), 1)
    context = {"value": 42}
    runtime = SimpleNamespace(
        state={"agents": {"thread_1": _agent("thread_1")}, "agent_runs": {}},
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
    assert "agents" not in command.update
    assert command.update["agent_runs"] == {
        "run_1": {
            "agent_run_id": "run_1",
            "agent_id": "thread_1",
            "run_id": "run_1",
            "status": "pending",
            "prompt": "do the task",
            "started_at": 100.0,
        }
    }
    assert json.loads(command.update["messages"][0].content) == {
        "agent_run_id": "run_1",
    }


@pytest.mark.parametrize(
    ("status", "collected", "expected_error"),
    [
        (None, False, "Unknown agent ID `run_1_thread`"),
        ("running", False, "active run `run_1`"),
        ("error", True, "status `\"error\"`"),
    ],
)
@pytest.mark.parametrize("tool_name", ["run", "fork"])
def test_agent_tools_reject_unavailable_agent(
    status: str | None, collected: bool, expected_error: str, tool_name: str
) -> None:
    client = MagicMock()
    tool = (
        _build_run_tool(_client_cache(client), 1)
        if tool_name == "run"
        else _build_fork_tool(_client_cache(client))
    )
    state = {"agents": {}, "agent_runs": {}}
    if status is not None:
        state["agents"]["run_1_thread"] = _agent("run_1_thread")
        run = _completed_agent_run("run_1") if collected else _agent_run("run_1")
        run["status"] = status
        state["agent_runs"]["run_1"] = run
    runtime = SimpleNamespace(state=state, context=None, tool_call_id="prompt_1")

    args = {"agent_id": "run_1_thread", "runtime": runtime}
    if tool_name == "run":
        args["prompt"] = "follow up"
    result = tool.func(**args)

    assert expected_error in result
    client.runs.create.assert_not_called()
    client.threads.create.assert_not_called()
    client.threads.copy.assert_not_called()


@pytest.mark.parametrize("async_invocation", [False, True])
def test_wait_stores_tool_calls_in_returned_response_order(
    monkeypatch, async_invocation: bool,
) -> None:
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(core.time, "time", lambda: clock.now)
    client = _completed_client(async_invocation)
    client.runs.get.side_effect = lambda thread_id, run_id: {
        "thread_id": thread_id,
        "run_id": run_id,
        "status": "running" if run_id == "run_d" else "success",
    }

    def get_history(thread_id, limit, metadata):
        clock.now += 10.0
        return _run_history(metadata["run_id"])

    client.threads.get_history.side_effect = get_history
    tool = _build_wait_tool(_client_cache(client))
    runtime = SimpleNamespace(
        state={
            "agents": {
                f"{run_id}_thread": _agent(f"{run_id}_thread")
                for run_id in ("run_c", "run_b", "run_a", "run_d")
            },
            "agent_runs": {
                "run_c": {
                    **_agent_run("run_c"),
                    "status": "responded",
                    "response": "earlier response",
                    "response_sequence_number": 0,
                    "collected_at": 3.0,
                },
                "run_b": _agent_run("run_b"),
                "run_a": _agent_run("run_a"),
                "run_d": {**_agent_run("run_d"), "status": "pending"},
            }
        },
        tool_call_id="wait_1",
    )

    if async_invocation:
        command = asyncio.run(tool.coroutine(runtime))
    else:
        command = tool.func(runtime)

    assert command.update["agent_runs"]["run_b"]["response_sequence_number"] == 1
    assert command.update["agent_runs"]["run_a"]["response_sequence_number"] == 2
    assert command.update["agent_runs"]["run_b"]["collected_at"] == 120.0
    assert command.update["agent_runs"]["run_a"]["collected_at"] == 120.0
    assert command.update["agent_runs"]["run_d"] == _agent_run("run_d")
    assert "run_c" not in command.update["agent_runs"]
    assert runtime.state["agent_runs"]["run_c"]["collected_at"] == 3.0
    assert command.update["agent_runs"]["run_b"]["tool_calls"] == [
        {"id": "run_b_call", "name": "resource_tool", "args": {"run": "run_b"}},
    ]
    assert command.update["agent_runs"]["run_a"]["tool_calls"] == [
        {"id": "run_a_call", "name": "resource_tool", "args": {"run": "run_a"}},
    ]
    assert json.loads(command.update["messages"][0].content) == [
        {
            "agent_id": f"{run_id}_thread",
            "agent_run_id": run_id,
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

    run = command.update["agent_runs"]["run_a"]
    assert run["status"] == "responded"
    assert run["response"] == ""
    assert json.loads(command.update["messages"][0].content)[0]["response"] == ""


@pytest.mark.parametrize("async_invocation", [False, True])
@pytest.mark.parametrize("status", ["error", "timeout", "interrupted"])
def test_wait_stores_tool_calls_from_failed_run(
    monkeypatch, async_invocation: bool, status: str,
) -> None:
    monkeypatch.setattr(core.time, "time", lambda: 200.0)
    client = _completed_client(async_invocation, status=status)
    tool = _build_wait_tool(_client_cache(client))
    runtime = _wait_runtime()

    if async_invocation:
        command = asyncio.run(tool.coroutine(runtime))
    else:
        command = tool.func(runtime)

    agent_run = command.update["agent_runs"]["run_a"]
    assert agent_run["status"] == status
    assert agent_run["collected_at"] == 200.0
    assert agent_run["tool_calls"] == [
        {"id": "run_a_call", "name": "resource_tool", "args": {"run": "run_a"}},
    ]
    assert agent_run["response"] is None
    if status == "error":
        assert json.loads(agent_run["error"]) == {
            "error": "RuntimeError", "message": "agent failed",
        }
    else:
        assert agent_run["error"] is None


@pytest.mark.parametrize("async_invocation", [False, True])
def test_wait_timeout_leaves_run_uncollected(monkeypatch, async_invocation: bool) -> None:
    monkeypatch.setattr(core, "WAIT_TIMEOUT_SECONDS", 0.0)
    client = _completed_client(async_invocation, status="running")
    tool = _build_wait_tool(_client_cache(client))
    runtime = _wait_runtime()
    runtime.state["agent_runs"]["run_a"]["status"] = "pending"

    if async_invocation:
        command = asyncio.run(tool.coroutine(runtime))
    else:
        command = tool.func(runtime)

    assert command.update["agent_runs"]["run_a"] == _agent_run("run_a")
    assert command.update["messages"][0].content == WAIT_TIMEOUT_ERROR
    client.threads.get_history.assert_not_called()


@pytest.mark.parametrize("async_invocation", [False, True])
def test_context_overflow_retires_worker_and_allows_replacement(async_invocation) -> None:
    client = _completed_client(async_invocation, status="error")
    context_error = {"error": "OpenAIContextOverflowError", "message": "128k context exceeded"}
    client.threads.get.return_value = {"error": context_error}
    cache = _client_cache(client)
    runtime = _wait_runtime()
    runtime.context = None
    tool = _build_wait_tool(cache)
    command = asyncio.run(tool.coroutine(runtime)) if async_invocation else tool.func(runtime)
    runtime.state["agent_runs"].update(command.update["agent_runs"])
    report = json.loads(command.update["messages"][0].content)[0]
    assert report["status"] == "error"
    assert json.loads(report["error"]) == context_error

    for tool, args in (
        (_build_run_tool(cache, 410), {"agent_id": "run_a_thread", "prompt": "try again"}),
        (_build_fork_tool(cache), {"agent_id": "run_a_thread"}),
    ):
        result = (asyncio.run(tool.coroutine(**args, runtime=runtime)) if async_invocation
                  else tool.func(**args, runtime=runtime))
        assert "status `\"error\"`" in result
    client.runs.create.assert_not_called()
    client.threads.copy.assert_not_called()

    client.threads.create.return_value = {"thread_id": "fresh_worker"}
    tool = _build_new_tool({AGENT_TYPE["agent_type_id"]: AGENT_TYPE}, cache)
    args = {"agent_type_id": AGENT_TYPE["agent_type_id"], "runtime": runtime}
    replacement = asyncio.run(tool.coroutine(**args)) if async_invocation else tool.func(**args)
    assert "fresh_worker" in replacement.update["agents"]
    client.threads.create.assert_called_once()


@pytest.mark.parametrize("async_invocation", [False, True])
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

    run = command.update["agent_runs"]["run_a"]
    assert run["status"] == "error"
    assert run["response"] is None
    assert run["error"] == expected_error
    assert run["response_sequence_number"] == 0
    assert json.loads(command.update["messages"][0].content) == [{
        "agent_id": "run_a_thread", "agent_run_id": "run_a",
        "status": "error", "response": None, "error": expected_error,
    }]
    runtime.state["agent_runs"].update(command.update["agent_runs"])
    assert core._get_current_agent_runs(runtime.state["agent_runs"]) == {}
    assert "status `\"error\"`" in core._get_agent_error("run_a_thread", runtime.state)
    middleware = DecomposerAgentMiddleware([AGENT_TYPE], 1)
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
    assert command.update["agent_runs"]["run_a"]["response"] == "response from run_a"


def test_decomposer_waits_before_creating_agents() -> None:
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
        agent_types=AGENT_TYPES,
    )
    inputs = {"messages": [{"role": "user", "content": "Hello"}]}

    result = agent.invoke(inputs)

    assert result["agents"] == {}
    assert result["agent_runs"] == {}
    assert any(
        isinstance(message, ToolMessage)
        and message.tool_call_id == "wait-call"
        and message.content == NO_ACTIVE_RUNS_ERROR
        for message in result["messages"]
    )
    assert result["messages"][-1].content == "done"


@pytest.mark.parametrize("async_invocation", [False, True])
def test_decomposer_delegates(monkeypatch, async_invocation: bool) -> None:
    monkeypatch.setattr(core.time, "time", lambda: 100.0)
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
                        _call("new", f"create-call-{i}", agent_type_id="dummy")
                        for i in range(2)
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        _call(
                            "run", f"prompt-call-{i}",
                            agent_id=f"thread_{i}", prompt="Say hello.",
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
        agent_types=AGENT_TYPES,
    )

    inputs = {"messages": [{"role": "user", "content": "Hello"}]}
    result = asyncio.run(agent.ainvoke(inputs)) if async_invocation else agent.invoke(inputs)

    assert client.runs.create.call_count == 2
    assert all(
        call.kwargs["config"] is None
        for call in client.runs.create.call_args_list
    )
    assert result["agents"] == {
        f"thread_{i}": {
            "agent_id": f"thread_{i}",
            "agent_type_id": "dummy",
            "assistant_id": "dummy",
            "thread_id": f"thread_{i}",
            "created_at": 100.0,
        }
        for i in range(2)
    }
    assert len(result["agent_runs"]) == 2
    for i in range(2):
        run = result["agent_runs"][f"thread_{i}_run"]
        assert run["agent_id"] == f"thread_{i}"
        assert run["prompt"] == "Say hello."
        assert run["started_at"] == 100.0
        assert run["collected_at"] == 100.0
        assert run["status"] == "responded"
        assert run["response"] == "hello"
        assert run["error"] is None
    responses = next(
        json.loads(message.content)
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "wait-call"
    )
    assert {response["agent_id"] for response in responses} == set(result["agents"])
    assert any(
        isinstance(message, HumanMessage) and message.content == EARLY_RESPONSE_ERROR
        for message in result["messages"]
    )
    assert result["messages"][-1].content == "dummy"


def test_decomposer_rejects_parallel_wait_and_finishes_with_idle_agent(monkeypatch) -> None:
    client = _mock_client(monkeypatch, False)
    client.threads.create.return_value = {"thread_id": "dummy-thread"}
    agent = create_decomposer_agent(
        decomposer_model=ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        _call("wait", "wait-call"),
                        _call("new", "create-call", agent_type_id="dummy"),
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        agent_types=AGENT_TYPES,
    )
    inputs = {"messages": [{"role": "user", "content": "Hello"}]}

    result = agent.invoke(inputs)

    client.threads.create.assert_called_once_with()
    client.runs.create.assert_not_called()
    client.runs.get.assert_not_called()
    assert list(result["agents"]) == ["dummy-thread"]
    assert result["agent_runs"] == {}
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
            AIMessage(content="", tool_calls=[_call("new", "create", agent_type_id="dummy")]),
            AIMessage(content="", tool_calls=[
                _call("wait", "wait_0"),
                _call("wait", "wait_1"),
                *[
                    _call("run", f"prompt_{i}", agent_id="thread_1", prompt=f"Task {i}")
                    for i in range(2)
                ],
            ]),
            AIMessage(content="done"),
        ]),
        agent_types=AGENT_TYPES,
    )
    inputs = {"messages": [{"role": "user", "content": "Hello"}]}

    result = agent.invoke(inputs)

    client.runs.create.assert_not_called()
    client.runs.get.assert_not_called()
    assert result["agent_runs"] == {}
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
        agent_types=AGENT_TYPES,
    )

    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})
    )

    assert any(
        isinstance(message, HumanMessage) and message.content == EMPTY_RESPONSE_ERROR
        for message in result["messages"]
    )
    assert result["messages"][-1].content == "done"


def _agent(agent_id: str) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "agent_type_id": "test",
        "assistant_id": "test_assistant",
        "thread_id": agent_id,
        "created_at": 1.0,
    }


def _agent_run(run_id: str) -> dict[str, Any]:
    return {
        "agent_run_id": run_id,
        "agent_id": f"{run_id}_thread",
        "run_id": run_id,
        "status": "running",
        "prompt": "do the task",
        "started_at": 2.0,
    }


def _completed_agent_run(run_id: str) -> dict[str, Any]:
    return {
        **_agent_run(run_id),
        "status": "responded",
        "response": "done",
        "response_sequence_number": 0,
        "collected_at": 3.0,
    }


def _client_cache(client):
    return SimpleNamespace(get_sync=lambda _: client, get_async=lambda _: client)


def _completed_client(async_invocation=False, status="success"):
    client = AsyncMock() if async_invocation else MagicMock()
    client.runs.get.side_effect = lambda thread_id, run_id: {
        "thread_id": thread_id, "run_id": run_id, "status": status,
    }
    client.threads.get.return_value = {
        "error": {"error": "RuntimeError", "message": "agent failed"},
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
            "agents": {"run_a_thread": _agent("run_a_thread")},
            "agent_runs": {"run_a": _agent_run("run_a")},
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
                _call("run", "run", agent_id="thread_1", prompt="Task"),
                _call("fork", "fork", agent_id="thread_1"),
            ]),
            AIMessage(content="done"),
        ]),
        agent_types=AGENT_TYPES,
    )
    inputs = {
        "messages": [HumanMessage("Hello")],
        "agents": {"thread_1": _agent("thread_1")},
    }
    result = agent.invoke(inputs)
    client.runs.create.assert_not_called()
    client.threads.copy.assert_not_called()
    assert [m.content for m in result["messages"] if isinstance(m, ToolMessage)] == [
        PARALLEL_FORK_RUN_CALL_ERROR, PARALLEL_FORK_RUN_CALL_ERROR,
    ]
