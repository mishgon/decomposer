import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from dummy_model import DummyModel
from langchain_core.messages import AIMessage

from decomposer import create_decomposer_agent

SUBAGENT_TYPES = [
    {
        "subagent_type_id": "dummy",
        "description": "Dummy subagent for smoke testing.",
        "assistant_id": "dummy",
        "url": "http://unused",
    }
]


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
    monkeypatch.setattr(
        "decomposer.core.get_sync_client",
        lambda **kwargs: client,
    )
    agent = create_decomposer_agent(
        DummyModel(
            tool_calls=[
                {
                    "name": "spawn_subagent",
                    "args": {
                        "subagent_type_id": "dummy",
                        "prompt": "Say hello.",
                    },
                    "id": "dummy-tool-call",
                    "type": "tool_call",
                }
            ]
        ),
        SUBAGENT_TYPES,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    client.runs.create.assert_called_once()
    assert result["subagent_runs"]["dummy-run"]["prompt"] == "Say hello."
    assert result["messages"][-1].content == "dummy"


def test_sync_retry_discards_invalid_response_and_reuses_history() -> None:
    model = DummyModel(
        responses=[
            _tool_message(
                _spawn_call("rejected-1", "Rejected one."),
                _spawn_call("rejected-2", "Rejected two."),
                usage=10,
            ),
            AIMessage(content="recovered", usage_metadata=_usage(4)),
        ]
    )
    agent = create_decomposer_agent(
        model,
        SUBAGENT_TYPES,
        max_tool_call_retries=1,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    assert result["messages"][-1].content == "recovered"
    assert len(result["messages"]) == 2
    assert model.seen_messages[0] == model.seen_messages[1]
    assert result["decomposer_retry_diagnostics"] == [
        {
            "outcome": "recovered",
            "attempts": 2,
            "max_retries": 1,
            "discarded_attempts": 1,
            "discarded_tool_call_counts": [2],
            "discarded_tool_names": [["spawn_subagent", "spawn_subagent"]],
            "discarded_usage": [_usage(10)],
        }
    ]
    assert "decomposer_failure" not in result


def test_default_has_no_retries_and_returns_structured_failure() -> None:
    model = DummyModel(
        responses=[
            _tool_message(
                _spawn_call("rejected-1", "Rejected one."),
                _spawn_call("rejected-2", "Rejected two."),
                usage=10,
            )
        ]
    )
    agent = create_decomposer_agent(model, SUBAGENT_TYPES)

    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    assert model.response_index == 1
    assert result["decomposer_failure"]["attempts"] == 1
    assert result["decomposer_failure"]["max_retries"] == 0
    assert result["decomposer_retry_diagnostics"][0]["discarded_attempts"] == 0


def test_async_retries_reset_each_turn_and_only_accepted_call_executes(monkeypatch) -> None:
    client = MagicMock()
    client.threads.create = AsyncMock(return_value={"thread_id": "accepted-thread"})
    client.runs.create = AsyncMock(return_value={"run_id": "accepted-run", "status": "pending"})
    client.runs.cancel = AsyncMock()
    monkeypatch.setattr("decomposer.core.get_client", lambda **kwargs: client)

    model = DummyModel(
        responses=[
            _tool_message(
                _spawn_call("turn-1-rejected-a", "Rejected A."),
                _spawn_call("turn-1-rejected-b", "Rejected B."),
                usage=11,
            ),
            _tool_message(_spawn_call("accepted", "Accepted."), usage=12),
            _tool_message(
                _spawn_call("turn-2-rejected", "Rejected C."),
                {"name": "wait", "args": {}, "id": "turn-2-wait", "type": "tool_call"},
                usage=13,
            ),
            AIMessage(content="done", usage_metadata=_usage(14)),
        ]
    )
    agent = create_decomposer_agent(
        model,
        SUBAGENT_TYPES,
        max_tool_call_retries=1,
    )

    result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]}))

    client.runs.create.assert_awaited_once()
    assert client.runs.create.await_args.kwargs["input"]["messages"][0]["content"] == "Accepted."
    client.runs.cancel.assert_not_awaited()
    assert model.seen_messages[0] == model.seen_messages[1]
    assert model.seen_messages[2] == model.seen_messages[3]
    assert [message.content for message in result["messages"] if isinstance(message, AIMessage)] == ["", "done"]
    assert [diagnostic["attempts"] for diagnostic in result["decomposer_retry_diagnostics"]] == [2, 2]
    assert [diagnostic["discarded_usage"] for diagnostic in result["decomposer_retry_diagnostics"]] == [
        [_usage(11)],
        [_usage(13)],
    ]
    assert "decomposer_failure" not in result


def test_async_exhaustion_retains_final_invalid_response_and_records_cancel_failure(monkeypatch) -> None:
    client = MagicMock()
    client.threads.create = AsyncMock(return_value={"thread_id": "accepted-thread"})
    client.runs.create = AsyncMock(return_value={"run_id": "accepted-run", "status": "pending"})
    client.runs.cancel = AsyncMock(side_effect=RuntimeError("cancel unavailable"))
    monkeypatch.setattr("decomposer.core.get_client", lambda **kwargs: client)

    final_invalid = _tool_message(
        _spawn_call("final-c", "Final C."),
        _spawn_call("final-d", "Final D."),
        usage=23,
    )
    model = DummyModel(
        responses=[
            _tool_message(_spawn_call("accepted", "Accepted."), usage=20),
            _tool_message(
                _spawn_call("rejected-a", "Rejected A."),
                _spawn_call("rejected-b", "Rejected B."),
                usage=21,
            ),
            _tool_message(
                _spawn_call("rejected-c", "Rejected C."),
                {"name": "wait", "args": {}, "id": "rejected-wait", "type": "tool_call"},
                usage=22,
            ),
            final_invalid,
        ]
    )
    agent = create_decomposer_agent(
        model,
        SUBAGENT_TYPES,
        max_tool_call_retries=2,
    )

    result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]}))

    client.runs.create.assert_awaited_once()
    client.runs.cancel.assert_awaited_once_with(
        thread_id="accepted-thread",
        run_id="accepted-run",
        wait=False,
        action="interrupt",
    )
    assert result["messages"][-1].tool_calls == final_invalid.tool_calls
    assert all(
        tool_call["id"] not in {"rejected-a", "rejected-b", "rejected-c", "rejected-wait"}
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    )
    assert result["decomposer_failure"] == {
        "class": "multiple_tool_calls_exhausted",
        "attempts": 3,
        "max_retries": 2,
        "tool_call_counts": [2, 2, 2],
        "tool_names": [
            ["spawn_subagent", "spawn_subagent"],
            ["spawn_subagent", "wait"],
            ["spawn_subagent", "spawn_subagent"],
        ],
        "subagent_cancellation": {
            "requested_run_ids": ["accepted-run"],
            "failures": [
                {
                    "subagent_run_id": "accepted-run",
                    "error_type": "RuntimeError",
                    "message": "cancel unavailable",
                }
            ],
        },
    }
    assert result["decomposer_retry_diagnostics"][-1] == {
        "outcome": "exhausted",
        "attempts": 3,
        "max_retries": 2,
        "discarded_attempts": 2,
        "discarded_tool_call_counts": [2, 2],
        "discarded_tool_names": [
            ["spawn_subagent", "spawn_subagent"],
            ["spawn_subagent", "wait"],
        ],
        "discarded_usage": [_usage(21), _usage(22)],
    }


@pytest.mark.parametrize("max_tool_call_retries", [-1, True, 1.5])
def test_rejects_invalid_max_tool_call_retries(max_tool_call_retries) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        create_decomposer_agent(
            DummyModel(),
            SUBAGENT_TYPES,
            max_tool_call_retries=max_tool_call_retries,
        )


def _usage(total: int) -> dict[str, int]:
    return {
        "input_tokens": total - 1,
        "output_tokens": 1,
        "total_tokens": total,
    }


def _spawn_call(call_id: str, prompt: str) -> dict:
    return {
        "name": "spawn_subagent",
        "args": {"subagent_type_id": "dummy", "prompt": prompt},
        "id": call_id,
        "type": "tool_call",
    }


def _tool_message(*tool_calls: dict, usage: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=list(tool_calls),
        usage_metadata=_usage(usage),
    )
