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


def test_last_budgeted_call_reserves_a_final_answer(tmp_path, monkeypatch):
    from langchain_core.messages import SystemMessage
    from gyms.wideseek import runtime

    tmp_path = tmp_path / 'episode'
    runtime.init_budget(tmp_path, calls=1, tokens=4096)
    monkeypatch.setattr(runtime, 'MODEL_BACKEND', 'local')
    monkeypatch.setattr(runtime, 'get_config', lambda: {'configurable': {'thread_id': 'test'}})
    request = SimpleNamespace(runtime=SimpleNamespace(context={'directory': str(tmp_path)}),
        model=SimpleNamespace(extra_body={'chat_template_kwargs': {'enable_thinking': False}}),
        model_settings={}, tools=[{'type': 'function', 'function': {
            'name': 'search', 'description': 'search', 'parameters': {'type': 'object'}}}],
        system_message=SystemMessage(content='Original instructions'),
        messages=[HumanMessage(content='Original question asks for a Markdown table')])
    request.override = lambda **kwargs: SimpleNamespace(**{**vars(request), **kwargs})
    seen = []

    async def handler(actual):
        seen.append(actual)
        return SimpleNamespace(result=[AIMessage(content='final answer')])

    asyncio.run(runtime.ModelLog('react', final_response=True).awrap_model_call(request, handler))
    assert seen[0].tool_choice == 'none'
    assert seen[0].tools == []
    assert 'tool_choice' not in seen[0].model_settings
    extra = seen[0].model_settings['extra_body']
    assert extra['chat_template_kwargs'] == {'enable_thinking': False}
    assert '<tool_call>' in extra['bad_words']
    assert 'regex' in extra['structured_outputs']
    assert 'Original instructions' in seen[0].system_message.text
    assert 'do not invent' in seen[0].system_message.text
    assert seen[0].messages == request.messages
    import json
    outcome = json.loads((tmp_path / 'model_calls.jsonl').read_text().splitlines()[-1])
    assert outcome['generation']['temperature'] == 0.
    assert outcome['generation']['extra_body']['structured_outputs'] == extra['structured_outputs']
    assert outcome['budget_finalization'] == 'local_markdown_grammar'
    assert outcome['budget_finalization_temperature'] == 0.
    assert outcome['effective_tool_count'] == 0
    assert 'No tool calls remain' in str(outcome['effective_system_message'])


def test_concurrent_reservations_identify_exactly_one_final_slot(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from gyms.wideseek.runtime import init_budget, reserve
    path = tmp_path / 'budget'
    init_budget(path, calls=2)
    with ThreadPoolExecutor(2) as pool:
        reservations = list(pool.map(lambda _: reserve(path, with_remaining=True), range(2)))
    assert sorted(remaining for _, remaining in reservations) == [0, 1]


def test_remote_researchers_do_not_force_the_manager_final_answer(tmp_path, monkeypatch):
    from gyms.wideseek import runtime
    runtime.init_budget(tmp_path / 'episode', calls=1)
    monkeypatch.setattr(runtime, 'get_config', lambda: {'configurable': {'thread_id': 'test'}})
    request = SimpleNamespace(runtime=SimpleNamespace(context={'directory': str(tmp_path / 'episode')}),
        model=SimpleNamespace(), model_settings={}, tools=[{'type': 'function', 'function': {
            'name': 'search', 'description': 'search', 'parameters': {'type': 'object'}}}],
        system_message=None, messages=[HumanMessage(content='question')])
    request.override = lambda **kwargs: SimpleNamespace(**{**vars(request), **kwargs})
    seen = []
    async def handler(actual):
        seen.append(actual)
        return SimpleNamespace(result=[AIMessage(content='result')])
    asyncio.run(runtime.ModelLog('researcher').awrap_model_call(request, handler))
    assert seen[0].tools == request.tools
    assert 'tool_choice' not in vars(seen[0])


def test_keyword_access_reaches_a_fact_beyond_the_preview(monkeypatch):
    text = 'irrelevant introduction. ' * 2000 + 'Enrollment in 2016 was 12345. ' + 'tail. ' * 1000
    monkeypatch.setattr(worker, 'request_tool', AsyncMock(return_value={'contents': text}))
    found = asyncio.run(worker.access.coroutine('url', None, access_token=1000, contains='2016'))
    assert 'Enrollment in 2016 was 12345.' in found
    assert len(found) < 1200
    missing = asyncio.run(worker.access.coroutine('url', None, contains='literal.*keyword'))
    assert 'not found' in missing


def test_final_answer_reservation_accounts_for_the_larger_token_allowance(tmp_path):
    import sqlite3
    from gyms.wideseek.runtime import init_budget, reserve
    path = tmp_path / 'budget'
    init_budget(path, calls=1, tokens=20000)
    limit, remaining = reserve(path, with_remaining=True, final_response=True)
    assert (limit, remaining) == (16384, 0)
    with sqlite3.connect(path / 'budget.sqlite') as db:
        assert db.execute('SELECT tokens FROM budget').fetchone()[0] == 3616


def test_timeout_only_generation_run_has_no_quality_score():
    from evals.wideseek.run import summarize
    rows = [{'evaluation': {'status': 'timeout', 'score': None},
             'started_at': 1, 'agent_finished_at': 2, 'status': 'timeout'}]
    summary = summarize(rows, skip_judge=True)
    assert summary['judge_skipped'] == 1
    assert summary['mean_native_score'] is None
    assert summary['mean_native_score_infra_zero'] is None
