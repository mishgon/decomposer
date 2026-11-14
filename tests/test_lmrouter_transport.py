import asyncio
import json
import ssl
from typing import get_args, get_type_hints

import httpx
import pytest

from decomposer import lmrouter_transport as recovery
from decomposer.models import create_model


@pytest.fixture
def delays(monkeypatch):
    values = []
    monkeypatch.setattr(recovery.random, 'uniform', lambda low, high: high)
    monkeypatch.setattr(recovery.time, 'sleep', values.append)
    async def sleep(value):
        values.append(value)
    monkeypatch.setattr(recovery.asyncio, 'sleep', sleep)
    return values


@pytest.mark.parametrize('asynchronous', [False, True])
def test_long_outage_recovers_same_request(asynchronous, delays, caplog):
    bodies = []
    responses = []
    def handler(request):
        bodies.append(request.content)
        if len(bodies) <= 3:
            raise httpx.ConnectError('secret URL must not enter logs')
        response = httpx.Response(503 if len(bodies) <= 12 else 200, json={'ok': True})
        responses.append(response)
        return response
    mock = httpx.MockTransport(handler)
    async def run():
        async with httpx.AsyncClient(transport=recovery.AsyncRecoveringTransport(mock)) as client:
            return await client.post('https://lmrouter.2a2i.org/v1/completions', json={'prompt': [1, 2]})
    if asynchronous:
        response = asyncio.run(run())
    else:
        with httpx.Client(transport=recovery.RecoveringTransport(mock)) as client:
            response = client.post('https://lmrouter.2a2i.org/v1/completions', json={'prompt': [1, 2]})
    assert response.json() == {'ok': True}
    assert len(bodies) == 13 and len(set(bodies)) == 1
    assert delays == [5, 10, 20, 40, 80, 160] + [300] * 6
    assert all(r.is_closed for r in responses)
    assert 'secret URL' not in caplog.text
    assert 'resumed after 12 retries' in caplog.text


@pytest.mark.parametrize('status', [400, 401, 403, 404, 422, 501])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_permanent_status_is_not_retried(status, asynchronous, delays):
    mock = httpx.MockTransport(lambda request: httpx.Response(status, json={'error': 'permanent'}))
    async def run():
        async with httpx.AsyncClient(transport=recovery.AsyncRecoveringTransport(mock)) as client:
            return await client.post('https://lmrouter.2a2i.org/v1/chat/completions', json={})
    if asynchronous:
        response = asyncio.run(run())
    else:
        with httpx.Client(transport=recovery.RecoveringTransport(mock)) as client:
            response = client.post('https://lmrouter.2a2i.org/v1/chat/completions', json={})
    assert response.status_code == status
    assert not delays


def test_cancellation_interrupts_recovery(monkeypatch):
    async def run():
        paused = asyncio.Event()
        async def sleep(delay):
            paused.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(recovery.asyncio, 'sleep', sleep)
        mock = httpx.MockTransport(lambda request: httpx.Response(503))
        async with httpx.AsyncClient(transport=recovery.AsyncRecoveringTransport(mock)) as client:
            task = asyncio.create_task(client.post('https://lmrouter.2a2i.org/v1/completions', json={}))
            await paused.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    asyncio.run(run())


def test_certificate_error_is_not_retried(delays):
    def handler(request):
        raise httpx.ConnectError('certificate') from ssl.SSLCertVerificationError('invalid certificate')
    with httpx.Client(transport=recovery.RecoveringTransport(httpx.MockTransport(handler))) as client:
        with pytest.raises(httpx.ConnectError):
            client.post('https://lmrouter.2a2i.org/v1/completions', json={})
    assert not delays


@pytest.mark.parametrize('url', ['https://other.test/v1/completions', 'https://lmrouter.2a2i.org/tool'])
def test_other_requests_are_not_replayed(url, delays):
    with httpx.Client(transport=recovery.RecoveringTransport(
            httpx.MockTransport(lambda request: httpx.Response(503)))) as client:
        assert client.post(url, json={}).status_code == 503
    assert not delays


@pytest.mark.parametrize('profile', [
    profile for profile in get_args(get_type_hints(create_model)['model_id'])
    if profile.startswith('lmrouter/')
])
def test_every_lmrouter_profile_uses_recovery(profile):
    model = create_model(profile)
    try:
        assert isinstance(model.http_client._transport, recovery.RecoveringTransport)
        assert isinstance(model.http_async_client._transport, recovery.AsyncRecoveringTransport)
        assert model.root_client.max_retries == model.root_async_client.max_retries == 0
    finally:
        model.http_client.close()
        asyncio.run(model.http_async_client.aclose())


def test_chat_and_direct_logprobs_use_same_recovery(delays):
    async def run():
        model = create_model('lmrouter/qwen_3_8_flash_next_non_thinking')
        requests = []
        def handler(request):
            requests.append(request)
            if len(requests) % 2:
                return httpx.Response(503)
            choice = {'index': 0, 'finish_reason': 'stop',
                      'message': {'role': 'assistant', 'content': 'OK'}}
            if request.url.path == '/v1/completions':
                choice = {'index': 0, 'text': 'OK', 'finish_reason': 'stop', 'prompt_logprobs': [None]}
            return httpx.Response(200, json={'id': 'test', 'created': 0, 'model': 'test',
                                           'object': 'chat.completion', 'choices': [choice]})
        await model.http_async_client._transport.aclose()
        model.http_async_client._transport = recovery.AsyncRecoveringTransport(httpx.MockTransport(handler))
        try:
            assert (await model.ainvoke('Hi')).content == 'OK'
            response = await model.root_async_client.completions.create(
                model=model.model_name, prompt=[42], max_tokens=1, extra_body={'prompt_logprobs': 0})
            assert response.choices[0].prompt_logprobs == [None]
            assert json.loads(requests[-1].content)['prompt'] == [42]
            assert len(requests) == 4
        finally:
            model.http_client.close()
            await model.http_async_client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('asynchronous', [False, True])
def test_partial_response_is_discarded_before_retry(asynchronous, delays):
    class BrokenBody(httpx.SyncByteStream, httpx.AsyncByteStream):
        closed = False
        def __iter__(self):
            yield b'{"choices":'
            raise httpx.ReadError('lost body')
        async def __aiter__(self):
            yield b'{"choices":'
            raise httpx.ReadError('lost body')
        def close(self):
            self.closed = True
        async def aclose(self):
            self.closed = True
    body = BrokenBody()
    attempts = []
    def handler(request):
        attempts.append(request)
        return httpx.Response(200, stream=body) if len(attempts) == 1 else httpx.Response(200, json={'ok': True})
    mock = httpx.MockTransport(handler)
    async def run():
        async with httpx.AsyncClient(transport=recovery.AsyncRecoveringTransport(mock)) as client:
            return await client.post('https://lmrouter.2a2i.org/v1/completions', json={})
    if asynchronous:
        response = asyncio.run(run())
    else:
        with httpx.Client(transport=recovery.RecoveringTransport(mock)) as client:
            response = client.post('https://lmrouter.2a2i.org/v1/completions', json={})
    assert response.json() == {'ok': True}
    assert body.closed and len(attempts) == 2
    assert delays == [5]


def test_backoff_has_jitter_and_a_cap(monkeypatch):
    bounds = []
    monkeypatch.setattr(recovery.random, 'uniform', lambda lo, hi: bounds.append((lo, hi)) or lo)
    assert recovery._delay(0, 'test') == 2.5
    assert recovery._delay(10000, 'test') == 150
    assert bounds == [(2.5, 5), (150, 300)]


@pytest.mark.parametrize('status', [408, 429, 500, 502, 503, 504])
def test_transient_status_retries(status, delays):
    statuses = iter([status, 200])
    with httpx.Client(transport=recovery.RecoveringTransport(
            httpx.MockTransport(lambda request: httpx.Response(next(statuses), json={})))) as client:
        assert client.post('https://lmrouter.2a2i.org/v1/completions', json={}).status_code == 200
    assert delays == [5]
