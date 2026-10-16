import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from gyms.synth.run import cancel_worker_run, cleanup_worker


@pytest.mark.parametrize("status", [404, 500])
def test_cancel_finished_worker_race(status):
    response = httpx.Response(status, request=httpx.Request("POST", "http://worker/cancel"))
    error = httpx.HTTPStatusError("cancel failed", request=response.request, response=response)
    client = SimpleNamespace(runs=SimpleNamespace(cancel=AsyncMock(side_effect=error)))
    if status == 404:
        asyncio.run(cancel_worker_run(client, "thread", "run"))
    else:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(cancel_worker_run(client, "thread", "run"))
    client.runs.cancel.assert_awaited_once_with("thread", "run", wait=True, action="interrupt")


def test_cleanup_retries_and_archives(tmp_path, monkeypatch):
    monkeypatch.setattr('gyms.synth.run.asyncio.sleep', AsyncMock())
    client = SimpleNamespace(
        runs=SimpleNamespace(list=AsyncMock(return_value=[])),
        threads=SimpleNamespace(get_state=AsyncMock(side_effect=[RuntimeError('500'), {}]),
                                delete=AsyncMock()))
    asyncio.run(cleanup_worker(client, 'thread', tmp_path))
    assert client.threads.get_state.await_count == 2
    client.threads.delete.assert_awaited_once()
    assert (tmp_path/'worker-thread.json').is_file()


def test_cleanup_exhaustion_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr('gyms.synth.run.asyncio.sleep', AsyncMock())
    client = SimpleNamespace(runs=SimpleNamespace(list=AsyncMock(side_effect=RuntimeError('500'))))
    with pytest.raises(RuntimeError):
        asyncio.run(cleanup_worker(client, 'thread', tmp_path))
    assert client.runs.list.await_count == 3
