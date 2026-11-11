import asyncio
import json
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from langchain_core.messages import AIMessage
import pytest

from gyms.widesearch import tools, run


def test_researcher_graph_accepts_tool_coroutines():
    from gyms.widesearch.agents import researcher_27b_thinking
    assert researcher_27b_thinking().get_graph()


def test_search_preserves_excerpts_and_limits_results(monkeypatch):
    request = AsyncMock(return_value={"results": [
        {"title": "A", "url": "https://a.test", "snippet": "Fact A\nFact B"},
        {"title": "B", "url": "https://b.test", "snippet": "Fact C"},
    ]})
    monkeypatch.setattr(tools, "request", request)
    result = asyncio.run(tools.search_global("query", count=1, use_english=True))
    assert "https://a.test" in result and "Fact B" in result
    assert "https://b.test" not in result
    assert request.call_args.args == ("search", {"query": "query", "max_results": 1, "mode": "pro"})


def test_page_reader_uses_description_and_saves_raw_model_output(monkeypatch, tmp_path):
    append = tools.append_record
    def append_outside_event_loop(*args):
        assert threading.current_thread() is not threading.main_thread()
        append(*args)
    monkeypatch.setattr(tools, "append_record", append_outside_event_loop)
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"content": "Source table"}))
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="Extracted fact")))
    monkeypatch.setattr(tools, "create_model", lambda _: model)
    result = asyncio.run(tools.text_browser_view("https://a.test", "Find fee"))
    assert result == "Extracted fact"
    prompt = model.ainvoke.call_args.args[0][-1]["content"]
    assert "Find fee" in prompt and "Source table" in prompt
    saved = json.loads((tmp_path / "web_calls.jsonl").read_text())
    assert saved["reader_model"] == tools.READER_MODEL


def test_tool_failure_is_visible_and_logged(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(side_effect=httpx.ReadTimeout("read timeout")))
    result = asyncio.run(tools.tools()[0].ainvoke({"query": "test"}))
    assert "ERROR: ReadTimeout" in result
    assert "read timeout" in (tmp_path / "web_calls.jsonl").read_text()


def test_web_runner_does_not_preflight_local_retrieval(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")
    main = AsyncMock()
    monkeypatch.setattr(run.browsecomp, "main", main)
    asyncio.run(run.main(SimpleNamespace(agent="researcher_27b_thinking", episode=False,
                                         retrieval_url=run.BACKEND)))
    assert main.call_args.kwargs["preflight_check"] is run.preflight
    with pytest.raises(ValueError, match="cannot resume"):
        asyncio.run(run.preflight("http://127.0.0.1:18082"))


def test_backend_requires_search_key(monkeypatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="KEENABLE_API_KEY"):
        asyncio.run(run.preflight(run.BACKEND))


def test_long_search_keeps_snippets_when_one_page_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"results": [
        {"url": "https://a.test", "snippet": "First snippet"},
        {"url": "https://b.test", "snippet": "Second snippet"},
    ]}))
    monkeypatch.setattr(tools, "read_page", AsyncMock(side_effect=[
        "Extracted first page", httpx.ReadTimeout("blocked page")]))
    result = asyncio.run(tools.tools()[0].ainvoke({"query": "test", "summary_type": "long"}))
    for text in ["First snippet", "Second snippet", "https://b.test", "Extracted first page", "blocked page"]:
        assert text in result


def test_reader_generation_error_becomes_tool_result(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"content": "Page text"}))
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("exhausted max_completion_tokens")))
    monkeypatch.setattr(tools, "create_model", lambda _: model)
    result = asyncio.run(tools.tools()[1].ainvoke({"url": "https://a.test", "description": "Find fee"}))
    assert "ERROR: RuntimeError" in result
    assert "max_completion_tokens" in (tmp_path / "web_calls.jsonl").read_text()


def test_long_search_keeps_results_when_reader_deadline_expires(monkeypatch):
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"results": [
        {"url": "https://a.test", "snippet": "Source snippet"},
    ]}))
    @asynccontextmanager
    async def expired(deadline):
        raise TimeoutError()
        yield
    monkeypatch.setattr(tools.asyncio, "timeout_at", expired)
    result = asyncio.run(tools.search_global("test", summary_type="long"))
    assert "Source snippet" in result
    assert "Reader time budget exhausted" in result


@pytest.mark.parametrize("base_url", [None, "http://127.0.0.1:18084/v1/"])
def test_rate_limit_retries_and_logs_without_key(monkeypatch, tmp_path, base_url):
    if base_url:
        monkeypatch.setenv("KEENABLE_API_URL", base_url)
    else:
        monkeypatch.delenv("KEENABLE_API_URL", raising=False)
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setenv("KEENABLE_API_KEY", "secret-test-key")
    attempts = []
    def respond(request):
        assert request.headers["x-api-key"] == "secret-test-key"
        assert request.method == "POST" and request.url.path == "/v1/search"
        assert str(request.url) == (base_url or tools.API_URL).rstrip("/") + "/search"
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"results": [{"url": "https://a.test", "snippet": "Fact"}],
                                        "usage": {"credits": 1, "paid": False}})
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    def make_client(**kwargs):
        client.headers.update(kwargs["headers"])
        return client
    monkeypatch.setattr(tools.httpx, "AsyncClient", make_client)
    sleep = AsyncMock()
    monkeypatch.setattr(tools.asyncio, "sleep", sleep)
    result = asyncio.run(tools.tools()[0].ainvoke({"query": "test", "count": 200}))
    assert "Fact" in result and len(attempts) == 2
    assert attempts[0]["max_results"] == 50
    assert 3 in [c.args[0] for c in sleep.call_args_list]
    raw = (tmp_path / "web_calls.jsonl").read_text()
    assert "secret-test-key" not in raw
    records = [json.loads(line) for line in raw.split("\n") if line]
    assert len([r for r in records if "status_code" in r]) == 2
    assert json.loads(records[-1]["response"])["usage"]["credits"] == 1


@pytest.mark.parametrize("status", [401, 402, 404])
def test_fetch_failure_does_not_retry_or_call_reader(monkeypatch, tmp_path, status):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")
    calls = []
    def respond(request):
        calls.append(request)
        assert request.method == "GET" and request.url.params["url"] == "https://a.test"
        return httpx.Response(status, json={"code": "failed"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kwargs: client)
    model = SimpleNamespace(ainvoke=AsyncMock())
    monkeypatch.setattr(tools, "create_model", lambda _: model)
    result = asyncio.run(tools.tools()[1].ainvoke({"url": "https://a.test", "description": "Fee"}))
    assert "HTTPStatusError" in result and str(status) in result
    assert len(calls) == 1
    model.ainvoke.assert_not_called()


@pytest.mark.parametrize("status,body,is_target_error", [
    (502, {"type": "target_error", "code": "target_unreachable"}, True),
    (502, {"type": "service_error", "code": "unavailable"}, False),
    (502, "Bad gateway", False),
    (401, {"type": "target_error", "code": "target_unreachable"}, False),
])
def test_target_page_failure_does_not_stop_queue(monkeypatch, tmp_path, status, body, is_target_error):
    from gyms.infrastructure import failures
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")
    attempts = []
    def respond(request):
        attempts.append(request)
        return httpx.Response(status, json=body) if isinstance(body, dict) else httpx.Response(status, text=body)
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(tools.asyncio, "sleep", AsyncMock())
    model = SimpleNamespace(ainvoke=AsyncMock())
    monkeypatch.setattr(tools, "create_model", lambda _: model)
    result = asyncio.run(tools.tools()[1].ainvoke({"url": "https://a.test", "description": "Find facts"}))
    assert len(attempts) == (4 if status == 502 else 1)
    assert ("Try another URL" in result) is is_target_error
    assert bool(failures(tmp_path)) is not is_target_error
    model.ainvoke.assert_not_awaited()
    records = [json.loads(line) for line in (tmp_path / "web_calls.jsonl").read_text().split("\n") if line]
    assert len([row for row in records if row.get("status_code") == status]) == len(attempts)
