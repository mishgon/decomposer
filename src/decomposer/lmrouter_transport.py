"""Recover non-streaming lmrouter inference requests after transient outages."""
import asyncio
import logging
import random
import ssl
import time

import httpx


log = logging.getLogger(__name__)
_STATUSES = {408, 429, 500, 502, 503, 504}
_ERRORS = (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError)


def _inference(request):
    return (request.url.host == 'lmrouter.2a2i.org' and request.method == 'POST'
            and request.url.path in {'/v1/chat/completions', '/v1/completions'})


def _check_tls(error):
    cause = error
    while cause is not None:
        if isinstance(cause, ssl.SSLCertVerificationError):
            raise error
        cause = cause.__cause__ or cause.__context__


def _delay(attempt, reason):
    ceiling = min(300, 5 * 2 ** min(attempt, 6))
    delay = random.uniform(ceiling / 2, ceiling)
    log.warning('lmrouter request paused: %s; retry %d in %.1fs', reason, attempt + 1, delay)
    return delay


class RecoveringTransport(httpx.BaseTransport):
    def __init__(self, transport):
        self.transport = transport

    def handle_request(self, request):
        if not _inference(request):
            return self.transport.handle_request(request)
        request.read()  # Buffer JSON so a failed request can be sent again unchanged.
        attempt = 0
        while True:
            response = None
            try:
                response = self.transport.handle_request(request)
                if response.status_code not in _STATUSES:
                    response.read()  # Do not expose a partial response to the SDK.
                    if attempt:
                        log.warning('lmrouter request resumed after %d retries', attempt)
                    return response
                reason = f'HTTP {response.status_code}'
            except _ERRORS as error:
                if response is not None and response.status_code >= 400 and response.status_code not in _STATUSES:
                    raise
                _check_tls(error)
                reason = type(error).__name__
            finally:
                if response is not None:
                    response.close()
            time.sleep(_delay(attempt, reason))
            attempt += 1

    def close(self):
        self.transport.close()


class AsyncRecoveringTransport(httpx.AsyncBaseTransport):
    def __init__(self, transport):
        self.transport = transport

    async def handle_async_request(self, request):
        if not _inference(request):
            return await self.transport.handle_async_request(request)
        await request.aread()
        attempt = 0
        while True:
            response = None
            try:
                response = await self.transport.handle_async_request(request)
                if response.status_code not in _STATUSES:
                    await response.aread()
                    if attempt:
                        log.warning('lmrouter request resumed after %d retries', attempt)
                    return response
                reason = f'HTTP {response.status_code}'
            except _ERRORS as error:
                if response is not None and response.status_code >= 400 and response.status_code not in _STATUSES:
                    raise
                _check_tls(error)
                reason = type(error).__name__
            finally:
                if response is not None:
                    await response.aclose()
            await asyncio.sleep(_delay(attempt, reason))
            attempt += 1

    async def aclose(self):
        await self.transport.aclose()
