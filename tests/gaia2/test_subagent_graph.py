from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

pytest.importorskip("langchain_openai")

from gyms.gaia2.subagents import graphs


TYPED_PARAMETERS = {
    "type": "object",
    "properties": {
        "age": {"type": "integer"},
        "recipients": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ],
            "default": None,
        },
    },
    "required": ["age"],
    "additionalProperties": False,
}

WAIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "SystemApp__wait_for_notification",
        "description": "Wait for the next notification.",
        "parameters": {
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "integer",
                    "default": 0,
                }
            },
            "additionalProperties": False,
        },
    },
}


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


def test_worker_argument_model_preserves_types_and_rejects_stringified_values():
    model = graphs._arguments_model("Typed", TYPED_PARAMETERS)

    assert model.model_validate(
        {"age": 24, "recipients": ["a@example.com"]}
    ).model_dump() == {
        "age": 24,
        "recipients": ["a@example.com"],
    }
    with pytest.raises(ValidationError):
        model.model_validate({"age": "24", "recipients": ["a@example.com"]})
    with pytest.raises(ValidationError):
        model.model_validate({"age": 24, "recipients": '["a@example.com"]'})
    with pytest.raises(ValidationError):
        model.model_validate({"age": 24, "unexpected": True})


def test_worker_returns_argument_validation_as_correctable_tool_feedback():
    schema = {
        "type": "function",
        "function": {
            "name": "Contacts__lookup",
            "description": "Look up contacts.",
            "parameters": TYPED_PARAMETERS,
        },
    }
    context = {
        "tool_schemas": [schema],
        "broker_url": "http://broker.test",
        "session_token": "token",
        "policy": "shared_serialized",
        "scenario_id": "scenario",
        "run_number": 1,
        "notification_cursor": 0,
    }
    tool = graphs._tool_from_schema(schema, context)

    result = asyncio.run(tool.ainvoke({"age": "24", "recipients": "[]"}))

    assert '"error": "Invalid tool arguments"' in result


def test_worker_wait_adapter_starts_from_context_and_advances_shared_cursor(
    monkeypatch,
):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/notifications/wait"):
            body = json.loads(request.content)
            assert body["cursor"] == 7
            assert body["arguments"] == {"timeout": 5}
            return httpx.Response(
                200,
                json={
                    "notifications": [{"sequence": 9, "message": "ready"}],
                    "next_cursor": 9,
                    "environment_stopped": False,
                    "native_wait_invoked": True,
                },
            )
        assert request.url.path.endswith("/notifications")
        assert request.url.params["cursor"] == "9"
        return httpx.Response(
            200,
            json={
                "notifications": [],
                "next_cursor": 9,
                "environment_stopped": False,
            },
        )

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        return async_client(transport=transport)

    monkeypatch.setattr(graphs.httpx, "AsyncClient", client_factory)
    context = {
        "tool_schemas": [WAIT_SCHEMA],
        "broker_url": "http://broker.test",
        "session_token": "token",
        "policy": "shared_serialized",
        "scenario_id": "scenario",
        "run_number": 1,
        "notification_cursor": 7,
    }
    tools = {
        tool.name: tool for tool in graphs._worker_tools(context, consumer="worker-a")
    }

    wait_result = asyncio.run(
        tools["SystemApp__wait_for_notification"].ainvoke({"timeout": 5})
    )
    read_result = asyncio.run(tools["Gaia2Broker__read_notifications"].ainvoke({}))

    assert '"native_wait_invoked": true' in wait_result
    assert '"next_cursor": 9' in read_result
    assert [request.url.path for request in requests] == [
        "/notifications/wait",
        "/notifications",
    ]


def test_worker_model_forwards_non_thinking_sampling(monkeypatch):
    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(graphs, "ChatOpenAI", fake_model)
    monkeypatch.setenv("GAIA2_SUBAGENT_MODEL", "worker")
    monkeypatch.setenv("GAIA2_SUBAGENT_ENDPOINT", "http://127.0.0.1:8023/v1")
    monkeypatch.setenv("GAIA2_SUBAGENT_TEMPERATURE", "0.7")
    monkeypatch.setenv("GAIA2_SUBAGENT_TOP_P", "0.8")
    monkeypatch.setenv("GAIA2_SUBAGENT_TOP_K", "20")
    monkeypatch.setenv("GAIA2_SUBAGENT_MIN_P", "0.0")
    monkeypatch.setenv("GAIA2_SUBAGENT_PRESENCE_PENALTY", "1.5")
    monkeypatch.setenv("GAIA2_SUBAGENT_REPETITION_PENALTY", "1.0")
    monkeypatch.setenv("GAIA2_SUBAGENT_MAX_COMPLETION_TOKENS", "4096")
    monkeypatch.setenv("GAIA2_SUBAGENT_THINKING", "0")
    monkeypatch.delenv("GAIA2_SUBAGENT_EXTRA_BODY", raising=False)

    graphs._model()

    assert captured["model"] == "worker"
    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.8
    assert captured["presence_penalty"] == 1.5
    assert captured["max_completion_tokens"] == 4096
    assert captured["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
