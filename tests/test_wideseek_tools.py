import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from gyms.wideseek import worker


def test_search_maps_count_and_bounds_unicode_passages(monkeypatch):
    request = AsyncMock(return_value=[{'url': 'long', 'contents': 'я' * 2000},
                                     {'url': 'short', 'contents': 'complete passage'}])
    monkeypatch.setattr(worker, 'request_tool', request)
    result = asyncio.run(worker.search.coroutine('question', None, count=7))
    request.assert_awaited_once_with('/retrieve', {'queries': ['question'], 'topk': 7}, None)
    assert result.startswith('# Search query:\nquestion\n# Result:\n')
    assert 'я' * 1500 in result and 'я' * 1501 not in result
    assert '[Doc 2](short):\ncomplete passage' in result


def test_access_summarizes_only_bounded_source_with_requested_intent(monkeypatch):
    content = 'я' * 6000 + 'MUST NOT REACH SUMMARIZER'
    monkeypatch.setattr(worker, 'request_tool', AsyncMock(return_value={'contents': content}))
    summarize = AsyncMock(return_value='Supported facts')
    monkeypatch.setattr(worker, 'summarize_access', summarize)
    result = asyncio.run(worker.access.coroutine('url', 'founding date', None))
    summarize.assert_awaited_once_with('founding date', 'я' * 6000, None)
    assert result == '# Access URL:\nurl\n# Info to extract:\nfounding date\n# Result:\nSupported facts'


@pytest.mark.parametrize('page', [None, {'contents': ''}, {'contents': ' \n'}, {'contents': None}])
def test_missing_pages_do_not_call_model(monkeypatch, page):
    monkeypatch.setattr(worker, 'request_tool', AsyncMock(return_value=page))
    summarize = AsyncMock()
    monkeypatch.setattr(worker, 'summarize_access', summarize)
    result = asyncio.run(worker.access.coroutine('url', 'intent', None))
    assert 'No useful Information is Found under this URL.' in result
    summarize.assert_not_called()


def test_model_schemas_match_single_and_batch_contracts():
    for tool, fields in [(worker.search, {'query', 'count'}),
                         (worker.access, {'url', 'info_to_extract'}),
                         (worker.search_batch, {'queries'}), (worker.access_batch, {'urls'})]:
        schema = convert_to_openai_tool(tool)['function']['parameters']
        assert set(schema['properties']) == fields
        assert 'runtime' not in schema['properties']
    with pytest.raises(ValidationError):
        worker.search.args_schema.model_validate({'query': 'x', 'count': 10, 'runtime': None})
    with pytest.raises(ValidationError):
        worker.access.args_schema.model_validate({'url': 'x', 'runtime': None})
    with pytest.raises(ValidationError):
        worker.access_batch.args_schema.model_validate({'urls': [
            {'url': 'x', 'info_to_extract': 'y'}] * 6, 'runtime': None})


def test_batches_execute_concurrently_and_preserve_input_order(monkeypatch):
    async def run():
        all_started = asyncio.Event()
        started = []

        async def request(endpoint, payload, runtime):
            query = payload['queries'][0]
            started.append(query)
            if len(started) == 2:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), 1)
            return [{'url': query, 'contents': query}]

        monkeypatch.setattr(worker, 'request_tool', request)
        result = await worker.search_batch.coroutine(
            [worker.SearchQuery(query='first'), worker.SearchQuery(query='second')], None)
        assert result.index('# Search query:\nfirst') < result.index('# Search query:\nsecond')

    asyncio.run(run())


@pytest.mark.parametrize('failure', [False, True])
def test_summary_logs_usage_and_failure_and_closes_model(tmp_path, monkeypatch, failure):
    response = AIMessage(content='Evidence', usage_metadata={
        'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110})
    policy = SimpleNamespace(ainvoke=AsyncMock(
        side_effect=RuntimeError('unavailable') if failure else None, return_value=response))
    monkeypatch.setattr(worker, 'model', lambda _: policy)
    close = AsyncMock()
    monkeypatch.setattr(worker, 'close_model', close)
    runtime = SimpleNamespace(context={'directory': str(tmp_path)}, config={'configurable': {'thread_id': 't'}})
    if failure:
        with pytest.raises(RuntimeError, match='unavailable'):
            asyncio.run(worker.summarize_access('date', 'Source', runtime))
    else:
        assert asyncio.run(worker.summarize_access('date', 'Source', runtime)) == 'Evidence'
    close.assert_awaited_once_with(policy)
    rows = [json.loads(line) for line in (tmp_path / 'model_calls.jsonl').read_text().splitlines()]
    assert rows[0]['role'] == 'access_summary'
    assert rows[0]['budget_scope'] == 'auxiliary'
    assert 'Source' in rows[0]['request_delta'][1]['data']['content']
    assert rows[1]['status'] == ('error' if failure else 'success')
    from gyms.wideseek.run import usage
    totals = usage(tmp_path)['access_summary']
    assert totals['calls'] == 1 and totals['errors'] == int(failure)
    assert totals['output_tokens'] == (0 if failure else 10)
