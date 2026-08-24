import asyncio
from unittest.mock import MagicMock

from decomposer import create_decomposer_agent
from decomposer.prompts import (
    EARLY_REPORT_ERROR,
    EMPTY_REPORT_ERROR,
    PARALLEL_WAIT_CALL_ERROR,
)
from dummy_model import DummyModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


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


def test_decomposer_creates() -> None:
    agent = create_decomposer_agent(DummyModel(), SUBAGENT_TYPES)

    assert agent is not None


def test_decomposer_runs() -> None:
    agent = create_decomposer_agent(DummyModel(), SUBAGENT_TYPES)

    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    assert result["messages"][-1].content == "dummy"


def test_decomposer_delegates(monkeypatch) -> None:
    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "dummy-thread"}
    client.runs.create.return_value = {
        "run_id": "dummy-run",
        "status": "pending",
    }
    client.runs.get.return_value = {
        "thread_id": "dummy-thread",
        "run_id": "dummy-run",
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
    monkeypatch.setattr(
        "decomposer.core.get_sync_client",
        lambda **kwargs: client,
    )
    agent = create_decomposer_agent(
        ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "spawn_subagent",
                            "args": {
                                "subagent_type_id": "dummy",
                                "prompt": "Say hello.",
                            },
                            "id": "spawn-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="early report"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "wait",
                            "args": {},
                            "id": "wait-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="dummy"),
            ]
        ),
        SUBAGENT_TYPES,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    client.runs.create.assert_called_once()
    assert result["subagent_runs"]["dummy-run"]["prompt"] == "Say hello."
    assert result["subagent_runs"]["dummy-run"]["report"]["content"] == "hello"
    assert any(
        isinstance(message, HumanMessage) and message.content == EARLY_REPORT_ERROR
        for message in result["messages"]
    )
    assert result["messages"][-1].content == "dummy"


def test_decomposer_rejects_parallel_wait_calls(monkeypatch) -> None:
    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "dummy-thread"}
    client.runs.create.return_value = {
        "run_id": "dummy-run",
        "status": "pending",
    }
    client.runs.get.return_value = {
        "thread_id": "dummy-thread",
        "run_id": "dummy-run",
        "status": "success",
    }
    input_message = {"type": "human", "content": "Do the task."}
    client.threads.get_history.return_value = [
        {
            "metadata": {"source": "loop"},
            "values": {
                "messages": [
                    input_message,
                    {"type": "ai", "content": "done", "tool_calls": []},
                ]
            },
        },
        {
            "metadata": {"source": "input"},
            "values": {"messages": [input_message]},
        },
    ]
    get_sync_client = MagicMock(return_value=client)
    monkeypatch.setattr("decomposer.core.get_sync_client", get_sync_client)
    agent = create_decomposer_agent(
        ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "wait",
                            "args": {},
                            "id": "wait-call",
                            "type": "tool_call",
                        },
                        {
                            "name": "spawn_subagent",
                            "args": {
                                "subagent_type_id": "dummy",
                                "prompt": "Do not run.",
                            },
                            "id": "spawn-call",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "wait",
                            "args": {},
                            "id": "second-wait-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        SUBAGENT_TYPES,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    get_sync_client.assert_called_once()
    client.runs.create.assert_called_once()
    client.runs.get.assert_called_once()
    assert result["subagent_runs"]["dummy-run"]["report"]["content"] == "done"
    rejected_calls = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.content == PARALLEL_WAIT_CALL_ERROR
    ]
    assert [message.tool_call_id for message in rejected_calls] == ["wait-call"]
    assert any(
        isinstance(message, ToolMessage) and message.tool_call_id == "spawn-call"
        for message in result["messages"]
    )
    assert result["messages"][-1].content == "done"


def test_decomposer_retries_empty_report_asynchronously() -> None:
    agent = create_decomposer_agent(
        ToolCallingFakeModel(
            responses=[
                AIMessage(content=" \n "),
                AIMessage(content="done"),
            ]
        ),
        SUBAGENT_TYPES,
    )

    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})
    )

    assert any(
        isinstance(message, HumanMessage) and message.content == EMPTY_REPORT_ERROR
        for message in result["messages"]
    )
    assert result["messages"][-1].content == "done"
