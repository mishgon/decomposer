import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gyms.wideseek.runtime import save
from sft.wideseek import launch
from sft.wideseek.outage import collect_episode, recoverable_cleanup


@pytest.mark.parametrize('error', ['thread: TimeoutError: ',
    'thread: RuntimeError: Worker still active after cancellation',
    'Episode cleanup exceeded 90s', 'thread: ReadError: disconnected', 'thread: PoolTimeout: '])
def test_only_worker_stop_failures_are_recoverable(error):
    assert recoverable_cleanup({'cleanup_errors': [error]})
    assert not recoverable_cleanup({'cleanup_errors': [error], 'export_error': 'disk failure'})
    assert not recoverable_cleanup({'cleanup_errors': ['thread: OSError: disk full']})


@pytest.mark.parametrize('cleanup_error', ['thread: TimeoutError: ', 'thread: PoolTimeout: '])
def test_restart_reaps_workers_before_requeue_and_preserves_results(tmp_path, monkeypatch, cleanup_error):
    events = []
    unsafe = tmp_path/'decomposer/bad/attempt-001/result.json'
    completed = tmp_path/'decomposer/good/attempt-001/result.json'
    save(completed, {'evaluation': {'score': 1}})
    (unsafe.parent/'execution-1').mkdir(parents=True)
    args = SimpleNamespace(output=tmp_path, worker_port=1, worker_slots=2,
                           worker_python=None, resume=False)

    @asynccontextmanager
    async def server(*a, **kw):
        events.append('start')
        try:
            yield 'http://worker'
        finally:
            assert unsafe.exists() if len(events) == 1 else not unsafe.exists()
            events.append('reaped')

    async def collect(args):
        if not args.resume:
            save(unsafe, {'execution_directory': 'execution-1',
                         'cleanup_errors': [cleanup_error]})
            await collect_episode(SimpleNamespace(wait=AsyncMock()), AsyncMock(),
                                  {'task_id': 'bad'}, 'decomposer', 1, tmp_path, args)
        assert events == ['start', 'reaped', 'start']
        assert completed.exists()
        assert not unsafe.exists()
        assert (unsafe.parent/'execution-1/cleanup-failed-result.json').exists()

    monkeypatch.setattr(launch, 'agent_server', server)
    monkeypatch.setattr(launch.asyncio, 'sleep', AsyncMock())
    asyncio.run(launch.run_with_workers(args, 'config', collect))
    assert events == ['start', 'reaped', 'start', 'reaped']


def test_unrelated_failures_are_not_restarted(tmp_path, monkeypatch):
    @asynccontextmanager
    async def server(*a, **kw):
        yield 'http://worker'
    async def collect(args):
        raise ValueError('invalid data')
    monkeypatch.setattr(launch, 'agent_server', server)
    args = SimpleNamespace(output=tmp_path, worker_port=1, worker_slots=2, worker_python=None)
    with pytest.raises(ValueError, match='invalid data'):
        asyncio.run(launch.run_with_workers(args, 'config', collect))


@pytest.mark.parametrize('soft,hard,expected', [
    (1024, 1048576, 65536), (1024, 4096, 4096),
    (1024, launch.resource.RLIM_INFINITY, 65536), (131072, 1048576, None),
])
def test_descriptor_limit_is_set_before_starting_workers(monkeypatch, soft, hard, expected):
    from unittest.mock import Mock
    set_limit = Mock()
    monkeypatch.setattr(launch.resource, 'getrlimit', lambda _: (soft, hard))
    monkeypatch.setattr(launch.resource, 'setrlimit', set_limit)
    @asynccontextmanager
    async def server(*args, **kwargs):
        if expected is None:
            set_limit.assert_not_called()
        else:
            set_limit.assert_called_once_with(launch.resource.RLIMIT_NOFILE, (expected, hard))
        yield 'http://worker'
    monkeypatch.setattr(launch, 'agent_server', server)
    args = SimpleNamespace(worker_port=1, worker_slots=512, worker_python=None)
    asyncio.run(launch.run_with_workers(args, 'config', AsyncMock()))
