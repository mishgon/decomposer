import asyncio
from unittest.mock import AsyncMock

import pytest

from opd.synth.retry import EpisodeTransportError, retry_episode


def test_long_outage_retries_until_recovery(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr('opd.synth.retry.asyncio.sleep', sleep)
    monkeypatch.setattr('opd.synth.retry.random.uniform', lambda lo, hi: hi)
    attempt = AsyncMock(side_effect=[EpisodeTransportError('saved')] * 12 + ['result'])
    assert asyncio.run(retry_episode(attempt)) == 'result'
    assert attempt.await_count == 13
    assert [c.args[0] for c in sleep.await_args_list] == [5,10,20,40,80,160] + [300]*6


def test_cancellation_still_stops_retries(monkeypatch):
    monkeypatch.setattr('opd.synth.retry.asyncio.sleep', AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(retry_episode(AsyncMock(side_effect=EpisodeTransportError('outage'))))


def test_non_transport_failure_is_not_retried():
    attempt = AsyncMock(side_effect=RuntimeError('bad model output'))
    with pytest.raises(RuntimeError):
        asyncio.run(retry_episode(attempt))
    assert attempt.await_count == 1


@pytest.mark.parametrize('result,retry', [
    ({'evaluation': {'score': None, 'errors': ['APIConnectionError: offline']}}, True),
    ({'error': 'WorkerError: offline', 'worker_errors': ['APIConnectionError: offline']}, True),
    ({'status': 'timeout', 'error': 'Episode exceeded 2700s'}, True),
    ({'error': 'ValueError: bad alignment'}, False),
])
def test_episode_outage_classification(result, retry):
    from opd.synth.retry import check_episode
    if retry:
        with pytest.raises(EpisodeTransportError): check_episode(result, has_tokens=False)
    else:
        check_episode(result, has_tokens=False)


@pytest.mark.parametrize('key', ['cleanup_errors', 'capture_error', 'trace_error', 'export_error'])
def test_unsafe_cleanup_is_never_retried(key):
    from opd.synth.retry import check_episode
    with pytest.raises(RuntimeError, match='cleanup/capture/export'):
        check_episode({key: 'failed', 'error': 'APIConnectionError: offline'}, has_tokens=True)


def test_mixed_permanent_error_is_fatal():
    from opd.synth.retry import check_episode
    with pytest.raises(RuntimeError, match='Mixed'):
        check_episode({'worker_errors': ['APIConnectionError: offline', 'AuthenticationError: bad key']}, has_tokens=True)
