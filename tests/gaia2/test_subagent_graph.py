from __future__ import annotations

import httpx
import pytest

pytest.importorskip("langchain_openai")

from gyms.gaia2.subagents import graphs


def test_worker_serializes_are_records_as_plain_json_tool_content():
    result = [
        {
            "name": "/Documents/wiki.md",
            "type": "file",
            "size": 42,
        }
    ]

    assert graphs._serialize_tool_result(result) == (
        '[{"name": "/Documents/wiki.md", "type": "file", "size": 42}]'
    )


def test_worker_preserves_text_tool_content():
    assert graphs._serialize_tool_result("already text") == "already text"


def test_worker_returns_model_correctable_broker_errors_as_tool_content():
    response = httpx.Response(
        400,
        json={"error": "unknown city"},
        request=httpx.Request("POST", "http://broker.test/invoke"),
    )

    assert graphs._broker_tool_result(response) == (
        '{"error": "unknown city", "http_status": 400}'
    )


def test_worker_does_not_hide_broker_authentication_errors():
    response = httpx.Response(
        401,
        json={"error": "invalid token"},
        request=httpx.Request("POST", "http://broker.test/invoke"),
    )

    with pytest.raises(httpx.HTTPStatusError):
        graphs._broker_tool_result(response)


def test_worker_model_forwards_non_thinking_sampling(monkeypatch):
    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(graphs, "ChatOpenAI", fake_model)
    monkeypatch.setenv("GAIA2_SUBAGENT_MODEL", "worker")
    monkeypatch.setenv("GAIA2_SUBAGENT_ENDPOINT", "http://127.0.0.1:8023/v1")
    monkeypatch.setenv("GAIA2_SUBAGENT_TEMPERATURE", "1.0")
    monkeypatch.setenv("GAIA2_SUBAGENT_TOP_P", "0.95")
    monkeypatch.setenv("GAIA2_SUBAGENT_MAX_COMPLETION_TOKENS", "4096")
    monkeypatch.setenv(
        "GAIA2_SUBAGENT_EXTRA_BODY",
        '{"top_k":64,"include_reasoning":false,'
        '"chat_template_kwargs":{"enable_thinking":false}}',
    )

    graphs._model()

    assert captured["model"] == "worker"
    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 0.95
    assert captured["max_completion_tokens"] == 4096
    assert captured["extra_body"] == {
        "top_k": 64,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
