import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import HumanMessage

from gyms.wideseek.run import cleanup_workers, episode
from gyms.wideseek.runtime import save


def client_with_runs(runs):
    async def list_runs(thread_id, *, limit, offset):
        return [dict(r) for r in runs[thread_id][offset:offset + limit]]

    async def cancel(thread_id, run_id, *, wait):
        assert wait
        next(r for r in runs[thread_id] if r['run_id'] == run_id)['status'] = 'interrupted'

    return SimpleNamespace(
        runs=SimpleNamespace(list=AsyncMock(side_effect=list_runs), cancel=AsyncMock(side_effect=cancel)),
        threads=SimpleNamespace(get_state=AsyncMock(return_value={'values': {'messages': [
            {'type': 'ai', 'content': 'answer', 'additional_kwargs': {'reasoning_content': 'reasoning'}}]}}),
            delete=AsyncMock()))


def state_for(*threads):
    return {'agents': {t: {'thread_id': t} for t in threads}, 'agent_runs': {}}


def test_archives_before_delete_and_releases_unused_threads(tmp_path):
    client = client_with_runs({'finished': [{'run_id': 'r1', 'status': 'success'}], 'unused': []})

    async def delete(thread_id):
        archived = json.loads((tmp_path / 'subagents' / f'{thread_id}.json').read_text())
        assert archived['values']['messages'][0]['additional_kwargs']['reasoning_content'] == 'reasoning'
        assert len(archived['runs']) == (1 if thread_id == 'finished' else 0)

    client.threads.delete.side_effect = delete
    assert asyncio.run(cleanup_workers(client, state_for('finished', 'unused'), tmp_path)) == []
    assert client.threads.delete.await_count == 2
    client.runs.cancel.assert_not_awaited()


def test_lost_launch_response_is_cleaned_before_evaluation(tmp_path):
    client = client_with_runs({'known-thread': [{'run_id': 'orphan', 'status': 'running'}]})
    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=httpx.ReadError('launch response lost'))
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={
        **state_for('known-thread'), 'messages': [HumanMessage(content='Question')]}))

    async def evaluate(*args, **kwargs):
        client.runs.cancel.assert_awaited_once_with('known-thread', 'orphan', wait=True)
        client.threads.delete.assert_awaited_once_with('known-thread')
        return {'status': 'scored', 'score': 0.}

    args = SimpleNamespace(agent='decomposer', model_calls=None, output_tokens=None, worker_url='http://unused', episode_timeout=1)
    task = {'task_id': 'task', 'question': 'Question', 'answer': 'Answer', 'unique_columns': []}
    with patch('gyms.wideseek.run.model', return_value=SimpleNamespace()), \
            patch('gyms.wideseek.run.agents.decomposer', return_value=graph), \
            patch('gyms.wideseek.run.get_client', return_value=client), \
            patch('gyms.wideseek.run.evaluate', new=evaluate):
        asyncio.run(episode(task, 'decomposer', 1, tmp_path, args))
    result = json.loads((tmp_path / 'decomposer/task/attempt-001/result.json').read_text())
    assert result['status'] == 'error'
    assert not result.get('cleanup_errors')
    archive = next(tmp_path.glob('decomposer/task/attempt-001/execution-*/subagents/*.json'))
    assert json.loads(archive.read_text())['runs'] == [{'run_id': 'orphan', 'status': 'interrupted'}]


def test_discovers_active_run_beyond_first_page(tmp_path):
    runs = [{'run_id': str(i), 'status': 'success'} for i in range(100)]
    runs.append({'run_id': 'late-orphan', 'status': 'pending'})
    client = client_with_runs({'thread': runs})
    assert asyncio.run(cleanup_workers(client, state_for('thread'), tmp_path)) == []
    client.runs.cancel.assert_awaited_once_with('thread', 'late-orphan', wait=True)
    assert len(json.loads((tmp_path / 'subagents/thread.json').read_text())['runs']) == 101


def test_failed_archive_retains_thread_and_cleans_other_threads(tmp_path):
    client = client_with_runs({'broken': [], 'healthy': []})

    def archive(path, value):
        if path.stem == 'broken':
            raise OSError('disk unavailable')
        save(path, value)

    with patch('gyms.wideseek.run.save', side_effect=archive):
        errors = asyncio.run(cleanup_workers(client, state_for('broken', 'healthy'), tmp_path))
    assert errors == ['broken: OSError: disk unavailable']
    client.threads.delete.assert_awaited_once_with('healthy')


def test_cancellation_failure_retains_thread(tmp_path):
    client = client_with_runs({'thread': [{'run_id': 'active', 'status': 'running'}]})
    client.runs.cancel.side_effect = TimeoutError('did not stop')
    errors = asyncio.run(cleanup_workers(client, state_for('thread'), tmp_path))
    assert errors == ['thread: TimeoutError: did not stop']
    client.threads.get_state.assert_not_awaited()
    client.threads.delete.assert_not_awaited()


def test_worker_still_running_after_cancel_is_not_deleted(tmp_path):
    client = client_with_runs({'thread': [{'run_id': 'active', 'status': 'running'}]})
    client.runs.cancel.side_effect = None
    errors = asyncio.run(cleanup_workers(client, state_for('thread'), tmp_path))
    assert errors == ['thread: RuntimeError: Worker still active after cancellation']
    client.threads.delete.assert_not_awaited()


def test_shared_server_cleanup_does_not_touch_another_episode(tmp_path):
    runs = {'episode-a': [{'run_id': 'a', 'status': 'running'}],
            'episode-b': [{'run_id': 'b', 'status': 'running'}]}
    client = client_with_runs(runs)
    assert asyncio.run(cleanup_workers(client, state_for('episode-a'), tmp_path)) == []
    assert runs['episode-b'][0]['status'] == 'running'
    client.threads.get_state.assert_awaited_once_with('episode-a')
    client.threads.delete.assert_awaited_once_with('episode-a')
    assert not (tmp_path / 'subagents/episode-b.json').exists()


def test_html_failure_preserves_result_and_resume_does_not_repeat(tmp_path):
    from langchain_core.messages import AIMessage
    from gyms.wideseek import run

    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={'messages': [AIMessage(content='Answer')]})
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={}))
    args = SimpleNamespace(agent='decomposer', model_calls=None, output_tokens=None,
                           worker_url='http://unused', episode_timeout=1)
    task = {'task_id': 'task', 'question': 'Question', 'unique_columns': []}
    result_path = tmp_path / 'decomposer/task/attempt-001/result.json'

    def render(*args):
        assert json.loads(result_path.read_text())['evaluation']['score'] == 1.
        raise OSError('HTML failed')

    with patch.object(run, 'model', return_value=SimpleNamespace()), \
            patch.object(run.agents, 'decomposer', return_value=graph), \
            patch.object(run, 'get_client', return_value=client_with_runs({})), \
            patch.object(run, 'evaluate', new=AsyncMock(return_value={'status': 'scored', 'score': 1.})), \
            patch.object(run, 'write_trace_html', side_effect=render):
        asyncio.run(episode(task, 'decomposer', 1, tmp_path, args))
        asyncio.run(episode(task, 'decomposer', 1, tmp_path, args))
    result = json.loads(result_path.read_text())
    assert result['status'] == 'finished'
    assert result['html_error'] == 'OSError: HTML failed'
    assert (tmp_path / result['trace_path']).is_file()
    graph.ainvoke.assert_awaited_once()


def test_snapshot_and_model_close_failures_still_archive_and_save(tmp_path):
    from langchain_core.messages import AIMessage
    from gyms.wideseek import run

    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={**state_for('thread'), 'messages': [AIMessage(content='Answer')]})
    graph.aget_state = AsyncMock(side_effect=RuntimeError('snapshot failed'))
    client = client_with_runs({'thread': []})
    args = SimpleNamespace(agent='decomposer', model_calls=None, output_tokens=None,
                           worker_url='http://unused', episode_timeout=1)
    with patch.object(run, 'model', return_value=SimpleNamespace()), \
            patch.object(run.agents, 'decomposer', return_value=graph), \
            patch.object(run, 'get_client', return_value=client), \
            patch.object(run, 'close_model', new=AsyncMock(side_effect=OSError('close failed'))), \
            patch.object(run, 'evaluate', new=AsyncMock(return_value={'status': 'scored', 'score': 1.})):
        asyncio.run(episode({'task_id': 'task', 'question': 'Question', 'unique_columns': []}, 'decomposer', 1, tmp_path, args))
    result = json.loads((tmp_path / 'decomposer/task/attempt-001/result.json').read_text())
    assert result['status'] == 'error'
    assert result['capture_error'] == 'RuntimeError: snapshot failed'
    assert result['cleanup_errors'] == ['Model cleanup: OSError: close failed']
    assert (tmp_path / result['trace_path']).is_file()
    client.threads.delete.assert_awaited_once_with('thread')


@pytest.mark.parametrize('stage', ['agent', 'cleanup', 'evaluation'])
def test_episode_deadline_cancels_each_stage_and_preserves_trace(tmp_path, stage):
    from langchain_core.messages import AIMessage
    from gyms.wideseek import run

    state = {'messages': [AIMessage(content='Partial answer')]}
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=state)
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(values=state))
    stopped = []

    async def block(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(stage)

    cleanup = AsyncMock(return_value=[])
    evaluate = AsyncMock(return_value={'status': 'scored', 'score': 1.})
    if stage == 'agent':
        graph.ainvoke.side_effect = block
    elif stage == 'evaluation':
        evaluate.side_effect = block
    else:
        async def cleanup_then_retry(*args, **kwargs):
            if not stopped:
                await block()
            return []
        cleanup.side_effect = cleanup_then_retry
    args = SimpleNamespace(agent='decomposer', model_calls=None, output_tokens=None,
                           worker_url='http://unused', episode_timeout=.05)
    with patch.object(run, 'model', return_value=SimpleNamespace()), \
            patch.object(run.agents, 'decomposer', return_value=graph), \
            patch.object(run, 'get_client', return_value=client_with_runs({})), \
            patch.object(run, 'cleanup_workers', cleanup), \
            patch.object(run, 'evaluate', evaluate):
        asyncio.run(episode({'task_id': 'task', 'question': 'Q', 'unique_columns': []},
                            'decomposer', 1, tmp_path, args))
    result = json.loads((tmp_path / 'decomposer/task/attempt-001/result.json').read_text())
    assert result['status'] == 'timeout'
    assert result['evaluation'] == {'status': 'timeout', 'score': None}
    assert json.loads((tmp_path / result['trace_path']).read_text())['messages'][0]['data']['content'] == 'Partial answer'
    assert stopped == [stage]
    assert cleanup.await_count == (2 if stage == 'cleanup' else 1)
    if stage != 'evaluation':
        evaluate.assert_not_awaited()
