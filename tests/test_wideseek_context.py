import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from gyms.wideseek import agents, worker


def test_page_slices_can_recover_every_character(monkeypatch):
    text = 'Wikipedia: αβγ 日本語\n' * 500
    monkeypatch.setattr(worker, 'request_tool', AsyncMock(return_value={'contents': text}))
    offset, slices = 0, []
    while True:
        result = asyncio.run(worker.access.coroutine('url', None, access_token=1000, offset=offset))
        slices.append(result.split('\n[Page truncated;')[0].removesuffix('\n[End of page.]'))
        match = re.search(r'next offset=(\d+)', result)
        if not match:
            break
        offset = int(match[1])
    assert ''.join(slices) == text
    assert asyncio.run(worker.access.coroutine('url', None, offset=len(text))) == '[End of page.]'


def test_search_marks_only_truncated_previews(monkeypatch):
    monkeypatch.setattr(worker, 'request_tool', AsyncMock(return_value=[
        {'url': 'long', 'contents': 'x' * 2000},
        {'url': 'short', 'contents': 'complete passage'},
    ]))
    result = asyncio.run(worker.search.coroutine('question', None))
    assert 'x' * 1500 in result and 'x' * 1501 not in result
    assert result.count('Passage preview truncated') == 1
    assert '[Doc 2](short):\ncomplete passage' in result


def test_context_edits_preserve_question_tool_pairs_and_raw_state(monkeypatch):
    monkeypatch.setattr(agents, 'MODEL_BACKEND', 'local')
    messages = [HumanMessage(content='Original question and required output columns')]
    for i in range(8):
        messages += [AIMessage(content='Verified fact with source URL', tool_calls=[
            {'name': 'search', 'args': {'query': str(i)}, 'id': str(i), 'type': 'tool_call'}]),
            ToolMessage(content=f'evidence {i} ' * 2000, tool_call_id=str(i))]
    request = SimpleNamespace(messages=messages)
    request.override = lambda **kwargs: SimpleNamespace(**{**vars(request), **kwargs})
    seen = []

    async def handler(edited):
        seen.extend(edited.messages)
        return AIMessage(content='answer')

    asyncio.run(agents.context_middleware()[0].awrap_model_call(request, handler))
    assert seen[0] == messages[0]
    old_results = [m for m in seen if isinstance(m, ToolMessage)][:-4]
    assert all('cleared from model context' in m.content for m in old_results)
    assert seen[-8:] == messages[-8:]
    assert all('cleared from model context' not in m.content for m in messages)
    assert [m.tool_calls for m in seen if isinstance(m, AIMessage)] == [
        m.tool_calls for m in messages if isinstance(m, AIMessage)]


def test_skip_judge_preserves_answer_and_marks_it_unscored(tmp_path, monkeypatch):
    import json
    from unittest.mock import MagicMock
    from gyms.wideseek import run

    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={'messages': [AIMessage(content='saved answer')]})
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={}))
    monkeypatch.setattr(run, 'model', lambda _: SimpleNamespace())
    monkeypatch.setattr(run.agents, 'react', lambda *args, **kwargs: graph)
    judge = AsyncMock(side_effect=AssertionError('Judge must not be called'))
    monkeypatch.setattr(run, 'evaluate', judge)
    args = run.create_parser().parse_args([
        '--agent', 'react', '--output', str(tmp_path), '--skip-judge'])
    task = {'task_id': 'smoke', 'question': 'question', 'answer': 'reference', 'unique_columns': []}
    asyncio.run(run.episode(task, 'simple', 1, tmp_path, args))
    saved = json.loads((tmp_path / 'simple/smoke/attempt-001/result.json').read_text())
    assert saved['answer'] == 'saved answer'
    assert saved['evaluation'] == {'status': 'not_requested', 'score': None}
    judge.assert_not_called()


def test_generation_only_summary_does_not_report_a_zero_quality_score():
    from evals.wideseek.run import summarize
    result = summarize([{'evaluation': {'status': 'not_requested', 'score': None},
                         'started_at': 1, 'agent_finished_at': 2, 'status': 'finished'}])
    assert result['judge_skipped'] == 1
    assert result['mean_native_score'] is None
    assert result['mean_native_score_infra_zero'] is None


def test_page_lookup_accepts_underscores_without_overriding_exact_urls(tmp_path):
    import json
    from gyms.wideseek.pages import Pages, build

    source = tmp_path / 'pages.jsonl'
    prefix = 'https://en.wikipedia.org/w/index.php?title='
    rows = [{'url': prefix + 'Alexander%20Fleming', 'contents': 'scientist'},
            {'url': prefix + 'Exact_Name', 'contents': 'exact spelling'},
            {'url': prefix + 'Exact%20Name', 'contents': 'space spelling'}]
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    build(source)
    pages = Pages(source)
    assert pages.get(prefix + 'Alexander_Fleming') == rows[0]
    assert pages.get(prefix + 'Exact_Name') == rows[1]
    assert pages.get(prefix + 'Missing_Page') is None
