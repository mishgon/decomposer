from unittest.mock import MagicMock

from dummy_model import DummyModel
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
