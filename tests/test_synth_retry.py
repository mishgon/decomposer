import asyncio
from unittest.mock import AsyncMock

import pytest

from opd.synth.retry import EpisodeTransportError, retry_episode


def test_transport_retry_is_bounded(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr('opd.synth.retry.asyncio.sleep', sleep)
    attempt = AsyncMock(side_effect=[EpisodeTransportError('saved first'),
                                    EpisodeTransportError('saved second'), 'result'])
    assert asyncio.run(retry_episode(attempt)) == 'result'
    assert attempt.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [1, 2]


def test_exhausted_transport_retries_fail(monkeypatch):
    monkeypatch.setattr('opd.synth.retry.asyncio.sleep', AsyncMock())
    attempt = AsyncMock(side_effect=EpisodeTransportError('outage'))
    with pytest.raises(EpisodeTransportError):
        asyncio.run(retry_episode(attempt))
    assert attempt.await_count == 3


def test_non_transport_failure_is_not_retried():
    attempt = AsyncMock(side_effect=RuntimeError('bad model output'))
    with pytest.raises(RuntimeError):
        asyncio.run(retry_episode(attempt))
    assert attempt.await_count == 1
