import asyncio
import time
from threading import Event
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config

import decomposer.run_budget as budget_module
from decomposer.prompts import AGENT_GRACEFUL_SHUTDOWN_REQUEST
from decomposer.run_budget import RunBudgetMiddleware


class FakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("asynchronous", [False, True])
def test_report_follows_all_tool_results_and_next_run_has_fresh_budget(monkeypatch, asynchronous):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(budget_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    requests = []

    class Capture(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            requests.append(request)
            return handler(request)

        async def awrap_model_call(self, request, handler):
            requests.append(request)
            return await handler(request)

    @tool(description="Perform part of the task.")
    def work(value: int) -> str:
        assert get_config()["configurable"]["thread_id"] == "test"
        clock.now += 6
        return str(value)

    agent = create_agent(
        model=FakeModel(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "work", "args": {"value": i}, "id": str(i)} for i in (1, 2)
            ]),
            AIMessage(content="Partial result"),
            AIMessage(content="Completed"),
        ]),
        system_prompt="Complete the task.",
        tools=[work], middleware=[RunBudgetMiddleware(), Capture()],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {
        "thread_id": "test", "agent_run_budget_seconds": 10,
        "agent_shutdown_grace_seconds": 30,
    }}

    async def run():
        inputs = {"messages": [HumanMessage(content="Do work")]}
        first = await agent.ainvoke(inputs, config) if asynchronous else agent.invoke(inputs, config)
        assert first["messages"][-1].content == "Partial result"
        assert first["messages"][-2].content == AGENT_GRACEFUL_SHUTDOWN_REQUEST
        assert all(isinstance(m, ToolMessage) for m in first["messages"][2:4])
        assert requests[0].tools
        assert requests[1].tools == []
        assert requests[1].response_format is None
        inputs = {"messages": [HumanMessage(content="Continue")]}
        second = await agent.ainvoke(inputs, config) if asynchronous else agent.invoke(inputs, config)
        assert second["messages"][-1].content == "Completed"
        assert requests[2].tools
        assert sum(m.content == AGENT_GRACEFUL_SHUTDOWN_REQUEST for m in second["messages"]) == 1
        assert second["shutdown_at"] == 22
        for request in requests:
            assert request.system_message.content.startswith("Complete the task.\n\n")
            assert request.system_message.content.count("You have up to 10 seconds") == 1
        assert not any("You have up to" in m.content for m in second["messages"])

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["model", "tool", "report"])
def test_hard_deadline_cancels_slow_operations(phase):
    cancelled = []

    class SlowModel(FakeModel):
        async def _agenerate(self, *args, **kwargs):
            if phase == "model" or (phase == "report" and self.i == 1):
                try:
                    await asyncio.sleep(10)
                finally:
                    cancelled.append("model")
            return await super()._agenerate(*args, **kwargs)

    @tool(description="Perform work.")
    async def work() -> str:
        if phase == "tool":
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append("tool")
        else:
            await asyncio.sleep(0.03)
        return "part"

    agent = create_agent(
        model=SlowModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "work", "args": {}, "id": "1"}]),
            AIMessage(content="Report"),
        ]), tools=[work], middleware=[RunBudgetMiddleware()],
    )

    async def run():
        async with asyncio.timeout(2):
            with pytest.raises(TimeoutError):
                await agent.ainvoke(
                    {"messages": [HumanMessage(content="Work")]},
                    {"configurable": {"agent_run_budget_seconds": .02, "agent_shutdown_grace_seconds": .1}},
                )
        assert cancelled == ["tool" if phase == "tool" else "model"]

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unlimited_agent_is_unchanged(asynchronous):
    agent = create_agent(model=FakeModel(responses=[AIMessage(content="Done")]), middleware=[RunBudgetMiddleware()])
    inputs = {"messages": [HumanMessage(content="Work")]}
    result = asyncio.run(agent.ainvoke(inputs)) if asynchronous else agent.invoke(inputs)
    assert len(result["messages"]) == 2
    assert result["messages"][-1].content == "Done"


def test_report_cannot_start_more_tools(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(budget_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    executed = []

    @tool(description="Perform work.")
    async def work() -> str:
        executed.append(True)
        clock.now = 2
        return "part"

    call = AIMessage(content="", tool_calls=[{"name": "work", "args": {}, "id": "1"}])
    agent = create_agent(model=FakeModel(responses=[call, call.model_copy(deep=True)]), tools=[work], middleware=[RunBudgetMiddleware()])
    with pytest.raises(ValueError, match="instead of returning its report"):
        asyncio.run(agent.ainvoke(
            {"messages": [HumanMessage(content="Work")]},
            {"configurable": {"agent_run_budget_seconds": 1, "agent_shutdown_grace_seconds": 30}},
        ))
    assert executed == [True]


@pytest.mark.parametrize("phase", ["model", "tool", "report"])
def test_sync_deadline_limits_waiting(phase):
    release = Event()
    finished = Event()

    class SlowModel(FakeModel):
        def _generate(self, *args, **kwargs):
            if phase == "model" or (phase == "report" and self.i == 1):
                release.wait(2)
                finished.set()
            return super()._generate(*args, **kwargs)

    @tool(description="Perform work.")
    def work() -> str:
        if phase == "tool":
            release.wait(2)
            finished.set()
        else:
            time.sleep(.03)
        return "part"

    agent = create_agent(
        model=SlowModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "work", "args": {}, "id": "1"}]),
            AIMessage(content="Report"),
        ]), tools=[work], middleware=[RunBudgetMiddleware()],
    )
    started_at = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            agent.invoke(
                {"messages": [HumanMessage(content="Work")]},
                {"configurable": {"agent_run_budget_seconds": .02, "agent_shutdown_grace_seconds": .1}},
            )
        assert time.monotonic() - started_at < 1
        assert not finished.is_set()
    finally:
        release.set()
        assert finished.wait(2)
