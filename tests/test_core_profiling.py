import asyncio
import logging
from unittest.mock import MagicMock

import pytest
from httpx import ReadError
from langsmith import Client, trace, tracing_context
from langchain_core.messages import AIMessage
from langchain_core.tracers.langchain import LangChainTracer, wait_for_all_tracers

import decomposer.core as core
from test_core import (
    AGENT_TYPES, ToolCallingFakeModel, _agent, _agent_run, _client_cache,
    _completed_client, _mock_client, _wait_runtime,
)


@pytest.fixture
def tracing_client(monkeypatch):
    client = Client(api_url="http://localhost:1", api_key="test", auto_batch_tracing=False)
    monkeypatch.setattr(Client, "create_run", MagicMock())
    monkeypatch.setattr(Client, "update_run", MagicMock())
    return client


@pytest.mark.parametrize("async_invocation", [False, True])
def test_wait_spans_join_parent_and_preserve_result(tracing_client, caplog, async_invocation):
    caplog.set_level(logging.INFO, logger=core.__name__)
    tool = core._build_wait_tool(_client_cache(_completed_client(async_invocation)))
    runtime = _wait_runtime()
    with tracing_context(enabled=True, client=tracing_client):
        with trace("wait", run_type="tool") as parent:
            result = asyncio.run(tool.coroutine(runtime)) if async_invocation else tool.func(runtime)
    assert result.update["agent_runs"]["run_a"]["status"] == "responded"
    spans = {span.name: span for span in parent.child_runs}
    assert {"threads.get_history", "wait.process_result", "wait.build_response"} <= spans.keys()
    poll = spans["wait.poll"].child_runs[0] if async_invocation else spans["runs.get"]
    assert poll.name == "runs.get"
    assert poll.metadata["agent_run_id"] == "run_a"
    assert poll.end_time >= poll.start_time
    assert all(span.parent_run_id == parent.id and span.end_time is not None for span in spans.values())
    assert "polls=1 active=1 collected=1" in caplog.text
    assert "history_entries=2 messages=3 tool_calls=1" in caplog.text
    assert "do the task" not in caplog.text


def test_parallel_poll_spans_overlap_and_keep_run_ids(tracing_client):
    runtime = _wait_runtime()
    runtime.state["agents"]["run_b_thread"] = _agent("run_b_thread")
    runtime.state["agent_runs"]["run_b"] = _agent_run("run_b")
    client = _completed_client(True)

    async def run():
        both_started = asyncio.Event()
        started = 0

        async def get(thread_id, run_id):
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            return {"thread_id": thread_id, "run_id": run_id, "status": "success"}

        client.runs.get.side_effect = get
        return await core._build_wait_tool(_client_cache(client)).coroutine(runtime)

    with tracing_context(enabled=True, client=tracing_client):
        with trace("wait", run_type="tool") as parent:
            result = asyncio.run(run())
    batch = next(span for span in parent.child_runs if span.name == "wait.poll")
    assert len(batch.child_runs) == 2
    assert {span.metadata["agent_run_id"] for span in batch.child_runs} == {"run_a", "run_b"}
    assert max(span.start_time for span in batch.child_runs) <= min(span.end_time for span in batch.child_runs)
    assert len(result.update["agent_runs"]) == 2


def test_transient_error_is_traced_and_retried(tracing_client, caplog, monkeypatch):
    monkeypatch.setattr(core, "WAIT_POLL_SECONDS", 0.0)
    caplog.set_level(logging.INFO, logger=core.__name__)
    client = _completed_client(True)
    client.runs.get.side_effect = [ReadError("temporary"), {
        "thread_id": "run_a_thread", "run_id": "run_a", "status": "success",
    }]
    tool = core._build_wait_tool(_client_cache(client))
    with tracing_context(enabled=True, client=tracing_client):
        with trace("wait", run_type="tool") as parent:
            result = asyncio.run(tool.coroutine(_wait_runtime()))
    batches = [span for span in parent.child_runs if span.name == "wait.poll"]
    assert "ReadError" in batches[0].child_runs[0].error
    assert batches[1].child_runs[0].error is None
    assert "wait.sleep" in [span.name for span in parent.child_runs]
    assert "retrying after ReadError" in caplog.text
    assert "polls=2 active=1 collected=1" in caplog.text
    assert result.update["agent_runs"]["run_a"]["status"] == "responded"


def test_cancelled_request_remains_cancelled(tracing_client):
    client = _completed_client(True)
    client.runs.get.side_effect = asyncio.CancelledError()
    with tracing_context(enabled=True, client=tracing_client):
        with trace("wait", run_type="tool") as parent:
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(core._build_wait_tool(_client_cache(client)).coroutine(_wait_runtime()))
    batch = next(span for span in parent.child_runs if span.name == "wait.poll")
    assert "CancelledError" in batch.child_runs[0].error


@pytest.mark.parametrize("async_invocation", [False, True])
def test_sdk_span_is_child_of_native_tool_run(tracing_client, monkeypatch, async_invocation):
    client = _mock_client(monkeypatch, async_invocation)
    client.threads.create.return_value = {"thread_id": "worker"}
    model = ToolCallingFakeModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "new", "args": {"agent_type_id": "dummy"}, "id": "new_1",
        }]),
        AIMessage(content="done"),
    ])
    agent = core.create_decomposer_agent(decomposer_model=model, agent_types=AGENT_TYPES)
    config = {"callbacks": [LangChainTracer(client=tracing_client)]}
    with tracing_context(enabled=True, client=tracing_client):
        if async_invocation:
            asyncio.run(agent.ainvoke({"messages": [("user", "request")]}, config))
        else:
            agent.invoke({"messages": [("user", "request")]}, config)
    wait_for_all_tracers()
    runs = [call.kwargs for call in tracing_client.create_run.call_args_list]
    sdk = next(run for run in runs if run["name"] == "threads.create")
    tool = next(run for run in runs if str(run["id"]) == str(sdk["parent_run_id"]))
    assert tool["name"] == "new"
    assert tool["run_type"] == "tool"
