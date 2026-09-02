from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

GAIA2_REPO = Path(__file__).parents[2] / "external" / "gaia2"
sys.path.insert(0, str(GAIA2_REPO))

from are.simulation.schema_reminders import (  # noqa: E402
    CANONICAL_TOOL_SCHEMA_REMINDER_PREFIX,
    render_openai_tool_retry_reminder,
)

pytest.importorskip("langchain_openai")

from gyms.gaia2.model_overflow import (  # noqa: E402
    ExactModelCallLimitMiddleware,
    Gaia2ModelOverflowMiddleware,
)
from gyms.gaia2.subagents import graphs  # noqa: E402

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

LOOKUP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Contacts__lookup",
        "description": "Look up contacts.",
        "parameters": TYPED_PARAMETERS,
    },
}


def test_worker_model_call_limit_is_read_from_runtime_environment(monkeypatch):
    monkeypatch.setenv("GAIA2_SUBAGENT_MAX_MODEL_CALLS", "200")
    assert graphs._max_model_calls() == 200
    monkeypatch.delenv("GAIA2_SUBAGENT_MAX_MODEL_CALLS")
    assert graphs._max_model_calls() is None


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

    result = graphs._broker_tool_result(response, LOOKUP_SCHEMA)
    metadata, reminder = result.split(
        "\n" + CANONICAL_TOOL_SCHEMA_REMINDER_PREFIX,
        1,
    )

    assert json.loads(metadata) == {"error": "unknown city", "http_status": 400}
    assert json.loads(reminder) == LOOKUP_SCHEMA
    assert result.endswith(
        render_openai_tool_retry_reminder(LOOKUP_SCHEMA).removeprefix(
            CANONICAL_TOOL_SCHEMA_REMINDER_PREFIX
        )
    )


def test_worker_does_not_hide_broker_authentication_errors():
    response = httpx.Response(
        401,
        json={"error": "invalid token"},
        request=httpx.Request("POST", "http://broker.test/invoke"),
    )

    with pytest.raises(httpx.HTTPStatusError):
        graphs._broker_tool_result(response)


def test_worker_preserves_canonical_json_schema_without_rebuilding_it():
    schema = LOOKUP_SCHEMA
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

    assert tool.args_schema == TYPED_PARAMETERS
    assert convert_to_openai_tool(tool) == schema


def test_worker_returns_broker_argument_validation_as_correctable_tool_feedback(
    monkeypatch,
):
    schema = LOOKUP_SCHEMA
    context = {
        "tool_schemas": [schema],
        "broker_url": "http://broker.test",
        "session_token": "token",
        "policy": "shared_serialized",
        "scenario_id": "scenario",
        "run_number": 1,
        "notification_cursor": 0,
    }
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            400,
            json={"error": "Argument 'age' must be of type int"},
        )

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        return async_client(transport=transport)

    monkeypatch.setattr(graphs.httpx, "AsyncClient", client_factory)
    tool = graphs._tool_from_schema(schema, context)
    result = asyncio.run(tool.ainvoke({"age": "24", "recipients": "[]"}))

    assert '"error": "Argument \'age\' must be of type int"' in result
    assert render_openai_tool_retry_reminder(schema) in result
    assert json.loads(requests[0].content) == {
        "arguments": {"age": "24", "recipients": "[]"}
    }


def test_worker_langchain_validation_errors_use_canonical_retry_reminder():
    context = {
        "tool_schemas": [LOOKUP_SCHEMA],
        "broker_url": "http://broker.test",
        "session_token": "token",
        "policy": "shared_serialized",
        "scenario_id": "scenario",
        "run_number": 1,
        "notification_cursor": 0,
    }
    tool = graphs._tool_from_schema(LOOKUP_SCHEMA, context)

    assert callable(tool.handle_validation_error)
    result = tool.handle_validation_error(ValueError("age must be an integer"))

    assert json.loads(result.split("\n", 1)[0]) == {
        "error": "age must be an integer"
    }
    assert render_openai_tool_retry_reminder(LOOKUP_SCHEMA) in result


def test_worker_model_requests_use_explicit_auto_tool_choice():
    class Request:
        def __init__(self, tools, tool_choice=None):
            self.tools = tools
            self.tool_choice = tool_choice

        def override(self, **values):
            return Request(
                self.tools,
                values.get("tool_choice", self.tool_choice),
            )

    captured = []

    async def handler(request):
        captured.append(request)
        return request

    middleware = graphs.Gaia2ToolChoiceAutoMiddleware()
    result = asyncio.run(middleware.awrap_model_call(Request([object()]), handler))

    assert result.tool_choice == "auto"
    assert captured[0].tool_choice == "auto"


def test_worker_model_requests_without_tools_are_unchanged():
    class Request:
        tools = []
        tool_choice = None

        def override(self, **_values):
            raise AssertionError("tool-free request must not be overridden")

    async def handler(request):
        return request

    request = Request()
    result = asyncio.run(
        graphs.Gaia2ToolChoiceAutoMiddleware().awrap_model_call(request, handler)
    )

    assert result is request


def test_worker_installs_fail_fast_overflow_middleware(monkeypatch):
    captured = {}

    class FakeAgent:
        async def ainvoke(self, value):
            return {"messages": value["messages"]}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr(graphs, "create_agent", fake_create_agent)
    monkeypatch.setattr(graphs, "_model", lambda: object())
    monkeypatch.setattr(graphs, "_worker_tools", lambda context, consumer: [])
    monkeypatch.setenv("GAIA2_SUBAGENT_MAX_MODEL_CALLS", "80")
    runtime = type("Runtime", (), {"context": {}})()

    asyncio.run(graphs.run_subagent({"messages": []}, runtime))

    assert any(
        isinstance(item, Gaia2ModelOverflowMiddleware)
        and item.actor == "subagent"
        and item.max_completion_tokens == 8192
        and item.max_model_len == 65536
        for item in captured["middleware"]
    )
    assert any(
        isinstance(item, ExactModelCallLimitMiddleware) and item.run_limit == 80
        for item in captured["middleware"]
    )


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
