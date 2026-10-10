import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

import gyms.agent_server as server


@pytest.fixture
def remote(monkeypatch):
    state = {
        "messages": [{"type": "human", "content": "Request"}],
        "decomposer_agent_runs": [{
            "agent_run_id": "root-run", "run_id": "root-run", "agent_id": "root",
            "status": "running", "prompt": "Request", "started_at": 1.0,
        }],
        "agent_runs": {"child-run": {
            "agent_run_id": "child-run", "run_id": "child-run", "agent_id": "child",
            "status": "running", "prompt": "Work", "started_at": 2.0,
        }},
    }
    statuses = {"root-run": "running", "child-run": "running"}
    client = MagicMock()
    client.__aenter__.return_value = client
    client.threads.create = AsyncMock(return_value={"thread_id": "root"})
    client.threads.search = AsyncMock(return_value=[{"thread_id": "root"}, {"thread_id": "child"}])
    client.threads.get_state = AsyncMock(side_effect=lambda _: {"values": deepcopy(state)})

    async def list_runs(thread_id, *, status, limit):
        run_id = f"{thread_id}-run"
        return [{"run_id": run_id}] if statuses[run_id] == status else []

    async def cancel(thread_id, run_id, *, wait, action):
        assert wait is True and action == "interrupt"
        statuses[run_id] = "interrupted"

    client.runs.list = AsyncMock(side_effect=list_runs)
    client.runs.cancel = AsyncMock(side_effect=cancel)
    client.runs.get = AsyncMock(side_effect=lambda thread_id, run_id: {"status": statuses[run_id]})
    monkeypatch.setattr(server, "get_client", lambda **kwargs: client)
    return client, state, statuses


@pytest.mark.parametrize("outcome", ["success", "error", "timeout", "cancelled"])
def test_capture_retains_checkpoint_and_confirmed_status(remote, outcome):
    client, checkpoint, statuses = remote
    failure = RuntimeError("model failed") if outcome == "error" else asyncio.CancelledError()

    async def wait(*args, **kwargs):
        if outcome == "success":
            statuses["root-run"] = "success"
            checkpoint["decomposer_agent_runs"][0].update(
                status="responded", response="Answer", collected_at=3.0,
            )
            return deepcopy(checkpoint)
        if outcome == "timeout":
            await asyncio.Event().wait()
        if outcome == "error":
            statuses["root-run"] = "error"
        raise failure

    client.runs.wait = AsyncMock(side_effect=wait)
    state, error = asyncio.run(server.invoke_and_capture(
        "http://server", "decomposer", {"messages": []}, timeout=0.05,
    ))
    root = state["decomposer_agent_runs"][0]
    assert state["messages"] == checkpoint["messages"]
    assert "collected_at" in root
    assert state["agent_runs"]["child-run"]["status"] == "interrupted"
    assert "collected_at" not in state["agent_runs"]["child-run"]
    assert checkpoint["agent_runs"]["child-run"]["status"] == "running"
    if outcome == "success":
        assert error is None
        assert root == checkpoint["decomposer_agent_runs"][0]
    else:
        assert root["status"] == ("error" if outcome == "error" else "interrupted")
        assert isinstance(error, TimeoutError) if outcome == "timeout" else error is failure
        assert root["error"] == state["agent_error"] == repr(error)
        assert "collected_at" not in checkpoint["decomposer_agent_runs"][0]


def test_unconfirmed_shutdown_keeps_run_open_and_original_error(remote):
    client, _, _ = remote
    failure = TimeoutError("agent deadline")
    client.runs.wait = AsyncMock(side_effect=failure)
    client.runs.cancel.side_effect = RuntimeError("server unavailable")
    state, error = asyncio.run(server.invoke_and_capture("http://server", "decomposer", {}))
    assert error is failure
    assert "server unavailable" in state["agent_shutdown_error"]
    root = state["decomposer_agent_runs"][0]
    assert root["status"] == "running"
    assert "collected_at" not in root


def test_capture_failure_preserves_original_model_error(remote):
    client, _, statuses = remote
    failure = RuntimeError("model failed")
    statuses["root-run"] = "error"
    client.runs.wait = AsyncMock(side_effect=failure)
    client.threads.get_state.side_effect = RuntimeError("checkpoint unavailable")
    state, error = asyncio.run(server.invoke_and_capture("http://server", "decomposer", {}))
    assert error is failure
    assert "checkpoint unavailable" in state["agent_capture_error"]
    assert state["agent_error"] == repr(failure)
