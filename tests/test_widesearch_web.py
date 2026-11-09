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


def test_search_pages_and_preserves_links(monkeypatch):
    request = AsyncMock(side_effect=[
        {"organic": [{"title": "A", "link": "https://a.test", "snippet": "Fact A"}]},
        {"organic": [{"title": "B", "link": "https://b.test", "snippet": "Fact B"}]},
    ])
    monkeypatch.setattr(tools, "request", request)
    result = asyncio.run(tools.search_global("query", count=150, use_english=True))
    assert "https://a.test" in result and "Fact B" in result
    assert [c.args[1]["page"] for c in request.call_args_list] == [1, 2]


def test_page_reader_uses_description_and_saves_raw_model_output(monkeypatch, tmp_path):
    append = tools.append_record
    def append_outside_event_loop(*args):
        assert threading.current_thread() is not threading.main_thread()
        append(*args)
    monkeypatch.setattr(tools, "append_record", append_outside_event_loop)
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"markdown": "Source table"}))
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


def test_missing_key_fails_before_launch(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SERPER_API_KEY"):
        asyncio.run(run.preflight(run.BACKEND))


def test_long_search_keeps_snippets_when_one_page_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"organic": [
        {"link": "https://a.test", "snippet": "First snippet"},
        {"link": "https://b.test", "snippet": "Second snippet"},
    ]}))
    monkeypatch.setattr(tools, "read_page", AsyncMock(side_effect=[
        "Extracted first page", httpx.ReadTimeout("blocked page")]))
    result = asyncio.run(tools.tools()[0].ainvoke({"query": "test", "summary_type": "long"}))
    for text in ["First snippet", "Second snippet", "https://b.test", "Extracted first page", "blocked page"]:
        assert text in result


def test_reader_generation_error_becomes_tool_result(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"text": "Page text"}))
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("exhausted max_completion_tokens")))
    monkeypatch.setattr(tools, "create_model", lambda _: model)
    result = asyncio.run(tools.tools()[1].ainvoke({"url": "https://a.test", "description": "Find fee"}))
    assert "ERROR: RuntimeError" in result
    assert "max_completion_tokens" in (tmp_path / "web_calls.jsonl").read_text()


def test_long_search_keeps_results_when_reader_deadline_expires(monkeypatch):
    monkeypatch.setattr(tools, "request", AsyncMock(return_value={"organic": [
        {"link": "https://a.test", "snippet": "Source snippet"},
    ]}))
    @asynccontextmanager
    async def expired(deadline):
        raise TimeoutError()
        yield
    monkeypatch.setattr(tools.asyncio, "timeout_at", expired)
    result = asyncio.run(tools.search_global("test", summary_type="long"))
    assert "Source snippet" in result
    assert "Reader time budget exhausted" in result


def test_provider_restriction_is_visible_without_logging_key(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    monkeypatch.setenv("SERPER_API_KEY", "secret-test-key")
    transport = httpx.MockTransport(lambda request: httpx.Response(
        400, json={"message": "Query pattern not allowed for free accounts."}))
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kwargs: client)
    result = asyncio.run(tools.tools()[0].ainvoke({"query": "test"}))
    assert "Query pattern not allowed for free accounts" in result
    assert "secret-test-key" not in (tmp_path / "web_calls.jsonl").read_text()
