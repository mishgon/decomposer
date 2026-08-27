from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from gyms.gaia2 import openrouter_proxy


def test_proxy_injects_credential_and_reasoning_and_retries_500(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.options = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def request(self, method, url, **kwargs):
            calls.append({"method": method, "url": url, **kwargs})
            request = httpx.Request(method, url)
            if len(calls) == 1:
                return httpx.Response(500, json={"error": "temporary"}, request=request)
            return httpx.Response(200, json={"choices": []}, request=request)

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(openrouter_proxy.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(openrouter_proxy.asyncio, "sleep", no_sleep)
    app = openrouter_proxy.create_app(
        upstream_url="https://openrouter.test/v1",
        api_key="openrouter-secret",
        extra_body={"reasoning": {"effort": "high"}},
        max_retries=2,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer judge-secret"},
            json={"model": "deepseek/model", "messages": []},
        )

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[-1]["url"] == "https://openrouter.test/v1/chat/completions"
    assert calls[-1]["headers"]["Authorization"] == "Bearer openrouter-secret"
    assert calls[-1]["json"] == {
        "model": "deepseek/model",
        "messages": [],
        "reasoning": {"effort": "high"},
    }


def test_proxy_retry_policy_excludes_non_retryable_client_errors() -> None:
    assert openrouter_proxy._retryable_status(429)
    assert openrouter_proxy._retryable_status(500)
    assert not openrouter_proxy._retryable_status(400)
    assert not openrouter_proxy._retryable_status(401)
