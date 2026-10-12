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

from decomposer.run_budget import RunBudgetMiddleware
from gyms.gaia2 import agents as graphs
from gyms.gaia2 import tools  # noqa: E402

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


def test_worker_serializes_are_records_as_plain_json_tool_content():
    result = [
        {
            "name": "/Documents/wiki.md",
            "type": "file",
            "size": 42,
        }
    ]

    assert tools._serialize_tool_result(result) == (
        '[{"name": "/Documents/wiki.md", "type": "file", "size": 42}]'
    )


def test_worker_preserves_text_tool_content():
    assert tools._serialize_tool_result("already text") == "already text"


def test_worker_returns_model_correctable_broker_errors_as_tool_content():
    response = httpx.Response(
        400,
        json={"error": "unknown city"},
        request=httpx.Request("POST", "http://broker.test/invoke"),
    )

    result = tools._broker_tool_result(response, LOOKUP_SCHEMA)
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
        tools._broker_tool_result(response)


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
    tool = tools._tool_from_schema(schema, context)

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

    monkeypatch.setattr(tools.httpx, "AsyncClient", client_factory)
    tool = tools._tool_from_schema(schema, context)
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
    tool = tools._tool_from_schema(LOOKUP_SCHEMA, context)

    assert callable(tool.handle_validation_error)
    result = tool.handle_validation_error(ValueError("age must be an integer"))

    assert json.loads(result.split("\n", 1)[0]) == {"error": "age must be an integer"}
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


def test_worker_uses_episode_prompt_budget_and_outer_checkpoint(monkeypatch):
    captured = {}

    class FakeAgent:
        async def ainvoke(self, value, *, context):
            captured["context"] = context
            return {"messages": value["messages"]}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr(graphs, "create_agent", fake_create_agent)
    monkeypatch.setattr(graphs, "_model", lambda: object())
    monkeypatch.setattr(graphs, "_worker_tools", lambda context, consumer: [])
    runtime = type(
        "Runtime", (), {"context": {"worker_system_prompt": "episode prompt"}}
    )()

    asyncio.run(graphs.run_agent({"messages": []}, runtime))

    assert any(isinstance(item, RunBudgetMiddleware) for item in captured["middleware"])
    # Forked workers only run if the per-run agent keeps out of the thread's
    # checkpoints (see run_agent).
    assert captured["checkpointer"] is False
    assert captured["system_prompt"] == "episode prompt"
    assert captured["context"] is runtime.context
    assert captured["context_schema"] is graphs.EpisodeContext


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

    monkeypatch.setattr(tools.httpx, "AsyncClient", client_factory)
    context = {
        "tool_schemas": [WAIT_SCHEMA],
        "broker_url": "http://broker.test",
        "session_token": "token",
        "policy": "shared_serialized",
        "scenario_id": "scenario",
        "run_number": 1,
        "notification_cursor": 7,
    }
    worker_tools = {
        tool.name: tool for tool in tools._worker_tools(context, consumer="worker-a")
    }

    wait_result = asyncio.run(
        worker_tools["SystemApp__wait_for_notification"].ainvoke({"timeout": 5})
    )
    read_result = asyncio.run(
        worker_tools["Gaia2Broker__read_notifications"].ainvoke({})
    )

    assert '"native_wait_invoked": true' in wait_result
    assert '"next_cursor": 9' in read_result
    assert [request.url.path for request in requests] == [
        "/notifications/wait",
        "/notifications",
    ]


def test_worker_uses_shared_model_factory(monkeypatch):
    monkeypatch.setenv("GAIA2_AGENT_MODEL_ID", "selected-model")
    monkeypatch.setenv("GAIA2_AGENT_BASE_URL", "http://agent/v1")
    sentinel = object()

    def factory(model_id, *, base_url):
        assert base_url == "http://agent/v1"
        assert model_id == "selected-model"
        return sentinel

    monkeypatch.setattr(graphs, "create_model", factory)
    assert graphs._model() is sentinel


@pytest.mark.parametrize("method", ["get_contact", "edit_contact"])
def test_real_contact_tool_schema_survives_worker_conversion(method):
    from are.simulation.apps.contacts import ContactsApp

    from gyms.gaia2.are import app_tool_schema

    native = next(
        tool for tool in ContactsApp().get_tools() if tool.name.endswith("__" + method)
    )
    schema = app_tool_schema(native)
    worker = tools._tool_from_schema(
        schema,
        {
            "broker_url": "http://broker.test",
            "session_token": "test",
        },
    )
    assert convert_to_openai_tool(worker) == schema
    assert worker.args_schema == schema["function"]["parameters"]
