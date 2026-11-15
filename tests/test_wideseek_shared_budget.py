import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from gyms.wideseek import runtime


def test_concurrent_workers_preserve_reports_and_coordinator_within_40_calls(tmp_path):
    path = tmp_path / 'episode'
    runtime.init_budget(path, calls=40, tokens=65536, coordinated=True)
    with ThreadPoolExecutor(4) as pool:
        admitted = list(pool.map(lambda i: runtime.register_worker(path, str(i)), range(4)))
    workers = [str(i) for i, ok in enumerate(admitted) if ok]
    assert len(workers) == 3
    records = []
    while workers:
        with ThreadPoolExecutor(3) as pool:
            batch = list(pool.map(lambda key: runtime.reserve(path, role='researcher', thread_id=key), workers))
        records.extend(batch)
        workers = [key for key, (_, _, final) in zip(workers, batch) if not final]
    assert not runtime.register_worker(path, 'late')
    coordinator = runtime.reserve(path, role='decomposer', final_response=True)
    assert coordinator[2] is True
    assert len(records) + 1 <= 40
    assert sum(final for _, _, final in records) == 3
    assert sum(limit for limit, _, _ in records) + coordinator[0] <= 65536
    with sqlite3.connect(path / 'budget.sqlite') as db:
        assert db.execute('SELECT calls,tokens FROM budget').fetchone()[0] >= 0
        assert db.execute('SELECT count(*) FROM workers').fetchone()[0] == 0


def test_call_cap_finalizes_even_when_tokens_are_unlimited(tmp_path):
    path = tmp_path / 'episode'
    runtime.init_budget(path, calls=40, coordinated=True)
    assert runtime.register_worker(path, 'worker')
    for _ in range(38):
        assert not runtime.reserve(path, role='researcher', thread_id='worker')[2]
    assert runtime.reserve(path, role='researcher', thread_id='worker')[2]
    assert runtime.reserve(path, role='decomposer', final_response=True)[2]
    with pytest.raises(runtime.BudgetExceeded):
        runtime.reserve(path, role='decomposer')


def test_waiting_does_not_consume_model_calls(tmp_path, monkeypatch):
    path = tmp_path / 'episode'
    runtime.init_budget(path, calls=40, coordinated=True)
    monkeypatch.setattr(runtime, 'get_config', lambda: {'configurable': {'thread_id': 'coordinator'}})
    request = SimpleNamespace(runtime=SimpleNamespace(context={'directory': str(path)}),
                              state={'agent_runs': {'worker': {'status': 'running'}}})
    handler = AsyncMock(side_effect=AssertionError('Waiting must not invoke a model'))
    result = asyncio.run(runtime.ModelLog('decomposer').awrap_model_call(request, handler))
    assert result.result[0].tool_calls[0]['name'] == 'wait'
    handler.assert_not_called()
    with sqlite3.connect(path / 'budget.sqlite') as db:
        assert db.execute('SELECT calls FROM budget').fetchone()[0] == 40
    assert not (path / 'model_calls.jsonl').exists()
    assert (path / 'budget_events.jsonl').exists()


class RepeatingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_researcher_returns_report_then_coordinator_can_answer(tmp_path):
    path = tmp_path / 'episode'
    runtime.init_budget(path, calls=4, coordinated=True)
    runtime.register_worker(path, 'worker')
    seen = []

    @tool(description='Search for evidence.')
    def search() -> str:
        return 'verified evidence'

    class Observe(runtime.AgentMiddleware):
        async def awrap_model_call(self, request, handler):
            seen.append(request)
            if not request.tools:
                return runtime.ModelResponse(result=[AIMessage(content='verified partial report')])
            return await handler(request)

    graph = create_agent(
        model=RepeatingModel(responses=[AIMessage(content='', tool_calls=[
            {'name': 'search', 'args': {}, 'id': 'search-call'}])]),
        tools=[search], middleware=[runtime.ModelLog('researcher'), Observe()],
    )
    result = asyncio.run(graph.ainvoke({'messages': [HumanMessage(content='Research')]},
        {'configurable': {'thread_id': 'worker'}}, context={'directory': str(path)}))
    assert result['messages'][-1].content == 'verified partial report'
    assert len(seen) == 3
    assert seen[-1].tools == []
    assert runtime.reserve(path, role='decomposer', final_response=True)[2]
