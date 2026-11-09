import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gyms.wideseek.runtime import save
from sft.wideseek.outage import OutageGuard, collect_episode, transient


@pytest.mark.parametrize('error', [
    'APIConnectionError: Connection error.', 'APITimeoutError: slow',
    'InternalServerError: nginx', 'HTTPStatusError: 503 Service Unavailable',
    'RateLimitError: Error code: 429', 'ReadError: disconnected',
])
def test_transient_errors(error):
    assert transient(error)


@pytest.mark.parametrize('error', [
    'AuthenticationError: Error code: 401', 'PermissionDeniedError: forbidden',
    'BadRequestError: Error code: 400', 'GraphRecursionError: limit',
    'Episode exceeded 2700s', 'ValueError: corrupt data',
])
def test_permanent_and_task_errors(error):
    assert not transient(error)


def test_shared_guard_survives_long_outage_with_capped_jitter(tmp_path, monkeypatch):
    delays = []
    async def sleep(delay):
        delays.append(delay)
    monkeypatch.setattr('sft.wideseek.outage.asyncio.sleep', sleep)
    monkeypatch.setattr('sft.wideseek.outage.random.uniform', lambda low, high: (low+high)/2)
    class APIConnectionError(Exception):
        pass
    probe = AsyncMock(side_effect=[*[APIConnectionError() for _ in range(20)], None])
    async def run():
        guard = OutageGuard(tmp_path, probe)
        guard.pause('outage')
        await asyncio.gather(*(guard.wait() for _ in range(128)))
        assert guard.since is None
    asyncio.run(run())
    assert probe.await_count == 21  # One recovery loop for all 128 workers.
    assert sum(delays) > 900
    assert max(delays) <= 300
    assert delays[:3] == [3.75, 7.5, 15]
    assert json.loads((tmp_path/'outage.json').read_text())['status'] == 'recovered'


def test_guard_permanent_probe_error_surfaces(tmp_path, monkeypatch):
    monkeypatch.setattr('sft.wideseek.outage.asyncio.sleep', AsyncMock())
    async def run():
        guard = OutageGuard(tmp_path, AsyncMock(side_effect=ValueError('bad configuration')))
        guard.pause('outage')
        with pytest.raises(ValueError):
            await guard.wait()
    asyncio.run(run())
    assert json.loads((tmp_path/'outage.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('source', ['coordinator', 'researcher', 'judge'])
def test_retry_same_attempt_and_preserve_raw_execution(tmp_path, monkeypatch, source):
    monkeypatch.setattr('sft.wideseek.outage.asyncio.sleep', AsyncMock())
    attempts = []
    result_path = tmp_path/'decomposer/task/attempt-001/result.json'
    async def episode(task, mode, attempt, root, args, **kwargs):
        attempts.append(attempt)
        execution = f'execution-{len(attempts)}'
        directory = result_path.parent/execution
        save(directory/'trace.json', {'messages': ['raw']})
        row = dict(execution_directory=execution, status='finished', evaluation=dict(status='scored', score=1))
        if len(attempts) == 1:
            error = 'APIConnectionError: connection failed'
            if source == 'coordinator':
                row.update(status='error', error=error)
            elif source == 'judge':
                row['evaluation'] = dict(status='judge_error', score=None, errors=[error])
            else:
                (directory/'model_calls.jsonl').write_text(json.dumps(dict(error=error))+'\n')
        save(result_path, row)
    async def run():
        guard = OutageGuard(tmp_path, AsyncMock())
        await collect_episode(guard, episode, dict(task_id='task'), 'decomposer', 1, tmp_path, SimpleNamespace())
    asyncio.run(run())
    assert attempts == [1, 1]
    assert json.loads(result_path.read_text())['evaluation']['score'] == 1
    archived = result_path.parent/'execution-1/infrastructure-result.json'
    assert json.loads(archived.read_text())['evaluation']['score'] is None
    assert (archived.parent/'trace.json').exists()


@pytest.mark.parametrize('row,raises', [
    (dict(status='timeout', evaluation=dict(status='timeout', score=None)), False),
    (dict(status='finished', evaluation=dict(status='scored', score=0)), False),
    (dict(status='error', error='ValueError: Decomposer requested tools instead of completing graceful shutdown'), False),
    (dict(status='error', cleanup_errors=['still active'], error='APIConnectionError'), True),
    (dict(status='error', error='AuthenticationError: invalid key'), True),
    (dict(status='finished', evaluation=dict(status='judge_error', errors=['ValueError: invalid response'])), True),
])
def test_task_outcomes_complete_but_unsafe_failures_surface(tmp_path, row, raises):
    result_path = tmp_path/'decomposer/task/attempt-001/result.json'
    async def episode(*args, **kwargs):
        save(result_path, dict(execution_directory='execution-1', **row))
    async def run():
        guard = OutageGuard(tmp_path, AsyncMock())
        await collect_episode(guard, episode, dict(task_id='task'), 'decomposer', 1, tmp_path, SimpleNamespace())
    if raises:
        with pytest.raises(RuntimeError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert result_path.exists()


def test_wait_cancellation_is_prompt(tmp_path):
    async def run():
        guard = OutageGuard(tmp_path, AsyncMock())
        guard.pause('outage')
        task = asyncio.create_task(guard.wait())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        guard.probe.assert_not_awaited()
    asyncio.run(run())


def test_batch_fatal_error_cancels_and_awaits_other_workers(tmp_path):
    from gyms.wideseek.run import run_jobs
    stopped = []
    async def run():
        started = asyncio.Event()
        async def runner(task, *args, **kwargs):
            if task['task_id'] == 'bad':
                await started.wait()
                raise RuntimeError('unsafe cleanup')
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(task['task_id'])
        save(tmp_path/'manifest.json', {'status': 'running'})
        args = SimpleNamespace(mode='decomposer', concurrency=2)
        with pytest.raises(RuntimeError, match='unsafe cleanup'):
            await run_jobs([dict(task_id='bad'), dict(task_id='other')],
                           [('bad', 1), ('other', 1)], tmp_path, args, episode_runner=runner)
    asyncio.run(run())
    assert stopped == ['other']
    assert json.loads((tmp_path/'manifest.json').read_text())['status'] == 'interrupted'


def test_watch_displays_pause(tmp_path, capsys):
    import fcntl
    import time
    from evals.wideseek.watch import display
    root = tmp_path/'run'
    save(root/'manifest.json', dict(started_at=time.time(), settings=dict(
        tasks=['task'], repetitions=1, modes=['simple'], concurrency=128, model='teacher', judge={})))
    save(root/'outage.json', dict(status='paused', since=time.time()-900, next_retry_at=time.time()+100))
    with (tmp_path/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        display(tmp_path)
    output = capsys.readouterr().out
    assert 'PAUSED: service unavailable | outage 0h 15m | next probe in' in output


def test_batch_keeps_task_failure_and_finishes_after_outage(tmp_path, monkeypatch):
    from functools import partial
    from gyms.wideseek.run import run_jobs
    monkeypatch.setattr('sft.wideseek.outage.asyncio.sleep', AsyncMock())
    calls = []
    async def episode(task, mode, attempt, root, args, **kwargs):
        task_id = task['task_id']
        calls.append((task_id, attempt))
        execution = f'execution-{len(calls)}'
        path = root/mode/task_id/f'attempt-{attempt:03d}'
        save(path/execution/'trace.json', {'messages': []})
        row = dict(episode_id=execution, task_id=task_id, task=task_id, mode=mode,
                   attempt=attempt, repetition=attempt, execution_directory=execution,
                   started_at=1, finished_at=2, trace_path=str(path/execution/'trace.json'),
                   status='finished', evaluation=dict(status='scored', score=1))
        if task_id == 'timeout':
            row.update(status='timeout', error='Episode exceeded 2700s',
                       evaluation=dict(status='timeout', score=None))
        elif calls.count((task_id, attempt)) == 1:
            row.update(status='error', error='APIConnectionError: offline',
                       evaluation=dict(status='scored', score=0))
        save(path/'result.json', row)
    async def run():
        save(tmp_path/'manifest.json', dict(status='running'))
        guard = OutageGuard(tmp_path, AsyncMock())
        await run_jobs([dict(task_id='timeout'), dict(task_id='retry')],
                       [('timeout', 1), ('retry', 1)], tmp_path,
                       SimpleNamespace(mode='decomposer', concurrency=2),
                       episode_runner=partial(collect_episode, guard, episode))
    asyncio.run(run())
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert len(manifest['episodes']) == 2
    assert {row['status'] for row in manifest['episodes']} == {'timeout', 'finished'}
    assert calls.count(('timeout', 1)) == 1
    assert calls.count(('retry', 1)) == 2
