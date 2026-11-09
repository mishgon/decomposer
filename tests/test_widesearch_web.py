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
        {"title": "A", "url": "https://a.test", "excerpts": ["Fact A", "Fact B"]},
        {"title": "B", "url": "https://b.test", "excerpts": ["Fact C"]},
    ]})
    monkeypatch.setattr(tools, "request", request)
    result = asyncio.run(tools.search_global("query", count=1, use_english=True))
    assert "https://a.test" in result and "Fact B" in result
    assert "https://b.test" not in result
    assert request.call_args.args == ("web_search", {"objective": "query", "search_queries": ["query"]})


def test_page_reader_uses_description_and_saves_raw_model_output(monkeypatch, tmp_path):
    append = tools.append_record
    def append_outside_event_loop(*args):
        assert threading.current_thread() is not threading.main_thread()
        append(*args)
    monkeypatch.setattr(tools, "append_record", append_outside_event_loop)
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"results": [{"excerpts": ["Source table"]}]}))
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
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    main = AsyncMock()
    monkeypatch.setattr(run.browsecomp, "main", main)
    asyncio.run(run.main(SimpleNamespace(agent="researcher_27b_thinking", episode=False,
                                         retrieval_url=run.BACKEND)))
    assert main.call_args.kwargs["preflight_check"] is run.preflight
    with pytest.raises(ValueError, match="cannot resume"):
        asyncio.run(run.preflight("http://127.0.0.1:18082"))


def test_free_backend_needs_no_search_key(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    asyncio.run(run.preflight(run.BACKEND))


def test_long_search_keeps_snippets_when_one_page_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"results": [
        {"url": "https://a.test", "excerpts": ["First snippet"]},
        {"url": "https://b.test", "excerpts": ["Second snippet"]},
    ]}))
    monkeypatch.setattr(tools, "read_page", AsyncMock(side_effect=[
        "Extracted first page", httpx.ReadTimeout("blocked page")]))
    result = asyncio.run(tools.tools()[0].ainvoke({"query": "test", "summary_type": "long"}))
    for text in ["First snippet", "Second snippet", "https://b.test", "Extracted first page", "blocked page"]:
        assert text in result


def test_reader_generation_error_becomes_tool_result(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"results": [{"content": "Page text"}]}))
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("exhausted max_completion_tokens")))
    monkeypatch.setattr(tools, "create_model", lambda _: model)
    result = asyncio.run(tools.tools()[1].ainvoke({"url": "https://a.test", "description": "Find fee"}))
    assert "ERROR: RuntimeError" in result
    assert "max_completion_tokens" in (tmp_path / "web_calls.jsonl").read_text()


def test_long_search_keeps_results_when_reader_deadline_expires(monkeypatch):
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"results": [
        {"url": "https://a.test", "excerpts": ["Source snippet"]},
    ]}))
    @asynccontextmanager
    async def expired(deadline):
        raise TimeoutError()
        yield
    monkeypatch.setattr(tools.asyncio, "timeout_at", expired)
    result = asyncio.run(tools.search_global("test", summary_type="long"))
    assert "Source snippet" in result
    assert "Reader time budget exhausted" in result


@pytest.mark.parametrize("mcp_error", [False, True])
def test_rate_limit_retries_and_logs_every_attempt(monkeypatch, tmp_path, mcp_error):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    attempts = []
    def respond(request):
        assert "x-api-key" not in request.headers
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            if mcp_error:
                return httpx.Response(200, json={"result": {"isError": True,
                    "content": [{"type": "text", "text": "Rate limit exceeded"}]}})
            return httpx.Response(429, headers={"Retry-After": "90"})
        return httpx.Response(200, json={"result": {"content": [{"type": "text", "text":
            json.dumps({"results": [{"url": "https://a.test", "excerpts": ["Fact"]}]})}]}})
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kwargs: client)
    sleep = AsyncMock()
    monkeypatch.setattr(tools.asyncio, "sleep", sleep)
    result = asyncio.run(tools.tools()[0].ainvoke({"query": "test"}))
    assert "Fact" in result and len(attempts) == 2
    assert (60 if mcp_error else 90) in [c.args[0] for c in sleep.call_args_list]
    records = [json.loads(line) for line in (tmp_path / "web_calls.jsonl").read_text().splitlines()]
    assert len([r for r in records if "status_code" in r]) == 2


def test_page_extraction_failure_does_not_call_reader(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"results": [], "errors": [
        {"url": "https://a.test", "error_type": "FETCH_FAILED"}]}))
    result = asyncio.run(tools.tools()[1].ainvoke({"url": "https://a.test", "description": "Fee"}))
    assert "FETCH_FAILED" in result
