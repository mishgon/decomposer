import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from evals.wideseek.band import resolved
from gyms.wideseek import run
from sft.wideseek.outage import OutageGuard, collect_episode


class OpenAIInvalidRequestError(Exception):
    pass


@pytest.mark.parametrize('arguments,status', [
    ('{"prompt": "unfinished', 400),
    ('{"prompt": "valid"}', 400),
    (None, 400),
    ('{"prompt": "unfinished', 401),
])
def test_only_proven_model_json_failure_allows_other_episodes(tmp_path, monkeypatch, arguments, status):
    malformed = '{"prompt": "unfinished'
    try:
        json.loads(malformed)
    except json.JSONDecodeError as exc:
        parser_error = str(exc)
    error = OpenAIInvalidRequestError(f'Error code: {status} - {parser_error}')
    messages = [HumanMessage('Question')]
    if arguments is not None:
        messages.append(AIMessage(content='', invalid_tool_calls=[
            {'id': 'bad-call', 'name': 'run', 'args': arguments, 'error': 'Invalid JSON'}]))
    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=[error, {'messages': [AIMessage(content='done')]}])
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={'messages': messages}))
    monkeypatch.setattr(run.agents, 'react', lambda *a, **kw: graph)
    policy = MagicMock()
    policy.http_async_client.aclose = AsyncMock()
    monkeypatch.setattr(run, 'model', lambda _: policy)
    monkeypatch.setattr(run, 'evaluate', AsyncMock(return_value={'status': 'scored', 'score': 0.0}))
    args = SimpleNamespace(agent='react', model_calls=None, output_tokens=None,
                           worker_url='http://unused', episode_timeout=10)
    tasks = [{'task_id': name, 'question': 'Question', 'answer': 'Answer', 'unique_columns': []}
             for name in ['malformed', 'healthy']]

    async def collect():
        guard = OutageGuard(tmp_path, AsyncMock())
        for task in tasks:
            await collect_episode(guard, run.episode, task, 'simple', 1, tmp_path, args)

    expected = arguments == malformed and status == 400
    if expected:
        asyncio.run(collect())
        healthy = json.loads((tmp_path / 'simple/healthy/attempt-001/result.json').read_text())
        assert healthy['status'] == 'finished'
    else:
        with pytest.raises(RuntimeError, match='Permanent service error'):
            asyncio.run(collect())
    attempt = tmp_path / 'simple/malformed/attempt-001'
    result = json.loads((attempt / 'result.json').read_text())
    assert result['status'] == ('invalid_tool_call' if expected else 'error')
    assert resolved(result, .9) is (False if expected else None)
    trace = json.loads((attempt / result['execution_directory'] / 'trace.json').read_text())
    assert trace['agent_error'] == result['error']
    if arguments is not None:
        assert trace['messages'][-1]['data']['invalid_tool_calls'][0]['args'] == arguments
