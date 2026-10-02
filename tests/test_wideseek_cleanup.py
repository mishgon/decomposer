import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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
    return {'subagents': {t: {'thread_id': t} for t in threads}, 'subagent_runs': {}}


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

    args = SimpleNamespace(model_calls=None, output_tokens=None, worker_url='http://unused', timeout=1)
    task = {'task_id': 'task', 'question': 'Question', 'answer': 'Answer', 'unique_columns': []}
    with patch('gyms.wideseek.run.model', return_value=SimpleNamespace()), \
            patch('gyms.wideseek.run.create_decomposer_agent', return_value=graph), \
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
