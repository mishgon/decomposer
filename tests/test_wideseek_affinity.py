import asyncio
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from gyms.wideseek import runtime
from gyms.wideseek.runtime import ModelLog, init_budget


@pytest.mark.parametrize('enabled', [False, True])
def test_affinity_is_stable_and_isolated_between_concurrent_threads(tmp_path, monkeypatch, enabled):
    assert runtime.SESSION_AFFINITY_ENABLED is True
    monkeypatch.setattr(runtime, 'SESSION_AFFINITY_ENABLED', enabled)
    current = ContextVar('thread')
    init_budget(tmp_path/'episode')
    seen = []
    shared = {'extra_headers': {'custom': 'keep'}}
    async def conversation(thread, hosted=True):
        current.set(thread)
        request = SimpleNamespace(runtime=SimpleNamespace(context={'directory': str(tmp_path/'episode')}),
            model=SimpleNamespace(openai_api_base='https://lmrouter.2a2i.org/v1' if hosted else 'http://localhost/v1'),
            model_settings=shared, messages=[HumanMessage(content='hi')], tools=[], system_message=None)
        request.override = lambda **kw: SimpleNamespace(**kw)
        async def handler(overridden):
            await asyncio.sleep(0)
            seen.append((thread, overridden.model_settings['extra_headers']))
            return SimpleNamespace(result=[AIMessage(content='OK')])
        for _ in range(2):
            await ModelLog('researcher').awrap_model_call(request, handler)
    async def run():
        await asyncio.gather(conversation('coordinator'), conversation('worker-a'),
                             conversation('worker-b'), conversation('local', False))
    with patch('gyms.wideseek.runtime.get_config', lambda: {'configurable': {'thread_id': current.get()}}):
        asyncio.run(run())
    assert len(seen) == 8
    for thread, headers in seen:
        assert headers['custom'] == 'keep'
        assert headers.get('x-litellm-session-id') == (thread if enabled and thread != 'local' else None)
    assert shared == {'extra_headers': {'custom': 'keep'}}
