import asyncio

import httpx
import pytest

from decomposer import lmrouter_transport as recovery
from opd.workplace_assistant import preflight


@pytest.mark.parametrize('permanent_status', [None, 400, 401, 403, 404])
def test_worker_preflight_recovers_outage_but_rejects_permanent_errors(monkeypatch, permanent_status):
    async def run():
        worker = preflight.create_model(preflight.WORKER_MODEL)
        monkeypatch.setattr(preflight, 'create_model', lambda name: worker)
        requests, delays = [], []
        real_sleep, real_wait_for = asyncio.sleep, asyncio.wait_for

        async def sleep(delay):
            delays.append(delay)
            await real_sleep(0.01)

        async def wait_for(awaitable, timeout):
            # Compress the old 30-second outer deadline so this regression is fast.
            return await real_wait_for(awaitable, 0.001 if timeout == 30 else timeout)

        monkeypatch.setattr(recovery.random, 'uniform', lambda low, high: high)
        monkeypatch.setattr(recovery.asyncio, 'sleep', sleep)
        monkeypatch.setattr(asyncio, 'wait_for', wait_for)

        def handler(request):
            requests.append(request)
            if permanent_status:
                return httpx.Response(permanent_status, json={'error': {'message': 'invalid configuration'}})
            if len(requests) == 1:
                raise httpx.ReadTimeout('slow inference', request=request)
            if len(requests) <= 5:
                return httpx.Response(503)
            return httpx.Response(200, json={
                'id': 'test', 'created': 0, 'model': 'test', 'object': 'chat.completion',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': 'OK'}}]})

        await worker.http_async_client._transport.aclose()
        worker.http_async_client._transport = recovery.AsyncRecoveringTransport(httpx.MockTransport(handler))
        if permanent_status:
            with pytest.raises(RuntimeError, match='Workplace validation worker preflight failed'):
                await preflight.check()
            assert len(requests) == 1
            assert delays == []
        else:
            await preflight.check()
            assert len(requests) == 6
            assert delays == [5, 10, 20, 40, 80]
            assert len({request.content for request in requests}) == 1
        assert worker.http_client.is_closed
        assert worker.http_async_client.is_closed

    asyncio.run(run())
