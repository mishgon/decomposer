import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx


GYM_DIR = Path(__file__).parents[1] / "external" / "Gym"
SUBAGENT_SERVER_DIR = (
    GYM_DIR
    / "responses_api_agents"
    / "decomposer_agent"
    / "subagent_server"
)
sys.path.insert(0, str(GYM_DIR))
sys.path.insert(0, str(SUBAGENT_SERVER_DIR))

import subagents  # noqa: E402
from subagents import (  # noqa: E402
    NeMoGymSubagentMiddleware,
    _to_chat_completions_tool,
    _to_chat_completions_tool_choice,
)


TOOLS = [
    {
        "type": "function",
        "name": "search_emails",
        "description": "Search emails.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": False,
    }
]


def test_converts_responses_tool_to_chat_completions_tool() -> None:
    assert _to_chat_completions_tool(TOOLS[0]) == {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Search emails.",
            "parameters": TOOLS[0]["parameters"],
            "strict": False,
        },
    }


def test_converts_responses_tool_choice_to_chat_completions_tool_choice() -> None:
    assert _to_chat_completions_tool_choice(
        {"type": "function", "name": "search_emails"}
    ) == {
        "type": "function",
        "function": {"name": "search_emails"},
    }
    for tool_choice in ("auto", "none", "required"):
        assert _to_chat_completions_tool_choice(tool_choice) == tool_choice


def test_model_hook_exposes_context_tools() -> None:
    middleware = NeMoGymSubagentMiddleware()
    request = _ModelRequest(_context())

    async def handler(updated_request: Any) -> Any:
        return updated_request

    updated_request = asyncio.run(middleware.awrap_model_call(request, handler))

    assert updated_request.tools == [_to_chat_completions_tool(TOOLS[0])]
    assert updated_request.tool_choice == "auto"
    assert updated_request.model_settings == {
        "existing": "setting",
        "parallel_tool_calls": False,
    }


def test_model_hook_converts_specific_tool_choice() -> None:
    middleware = NeMoGymSubagentMiddleware()
    context = _context()
    context["body"]["tool_choice"] = {
        "type": "function",
        "name": "search_emails",
    }
    request = _ModelRequest(context)

    async def handler(updated_request: Any) -> Any:
        return updated_request

    updated_request = asyncio.run(middleware.awrap_model_call(request, handler))

    assert updated_request.tool_choice == {
        "type": "function",
        "function": {"name": "search_emails"},
    }


def test_model_hook_applies_responses_api_defaults() -> None:
    middleware = NeMoGymSubagentMiddleware()
    context = _context()
    del context["body"]["parallel_tool_calls"]
    request = _ModelRequest(context)

    async def handler(updated_request: Any) -> Any:
        return updated_request

    updated_request = asyncio.run(middleware.awrap_model_call(request, handler))

    assert updated_request.tool_choice == "auto"
    assert updated_request.model_settings["parallel_tool_calls"] is True


def test_model_hook_supports_context_without_tools() -> None:
    middleware = NeMoGymSubagentMiddleware()
    context = _context()
    del context["body"]["tools"]
    request = _ModelRequest(context)

    async def handler(updated_request: Any) -> Any:
        return updated_request

    updated_request = asyncio.run(middleware.awrap_model_call(request, handler))

    assert updated_request.tools == []
    assert updated_request.tool_choice is None
    assert updated_request.model_settings == {"existing": "setting"}


def test_tool_hook_calls_resource_server(monkeypatch) -> None:
    fake_client = _FakeAsyncClient()

    def client_factory(**kwargs: Any) -> "_FakeAsyncClient":
        fake_client.timeout = kwargs["timeout"]
        return fake_client

    monkeypatch.setattr(subagents, "AsyncClient", client_factory)
    context = _context()
    request = SimpleNamespace(
        runtime=SimpleNamespace(context=context),
        tool_call={
            "name": "search_emails",
            "args": {"query": "quarterly report"},
            "id": "call_1",
        },
    )

    result = asyncio.run(
        NeMoGymSubagentMiddleware().awrap_tool_call(
            request,
            _unexpected_tool_handler,
        )
    )

    assert fake_client.timeout == 300.0
    assert fake_client.post_args == (
        "http://resources.test/search_emails",
        {"query": "quarterly report"},
        {"session": "seeded"},
    )
    assert result.content == '{"output":"found"}'
    assert result.tool_call_id == "call_1"
    assert result.status == "success"
    assert context["resource_server_cookies"] == {
        "session": "seeded",
        "refreshed": "cookie",
    }


def test_tool_hook_rejects_unknown_tool_without_http_request(monkeypatch) -> None:
    def unexpected_client(**kwargs: Any) -> Any:
        raise AssertionError("HTTP client must not be created for unknown tools")

    monkeypatch.setattr(subagents, "AsyncClient", unexpected_client)
    request = SimpleNamespace(
        runtime=SimpleNamespace(context=_context()),
        tool_call={"name": "delete_everything", "args": {}, "id": "call_2"},
    )

    result = asyncio.run(
        NeMoGymSubagentMiddleware().awrap_tool_call(
            request,
            _unexpected_tool_handler,
        )
    )

    assert result.status == "error"
    assert result.content == (
        "Error: delete_everything is not a valid tool, try one of [search_emails]."
    )


def _context() -> dict[str, Any]:
    return {
        "body": {
            "tools": TOOLS,
            "parallel_tool_calls": False,
        },
        "resource_server_url": "http://resources.test",
        "resource_server_cookies": {"session": "seeded"},
    }


async def _unexpected_tool_handler(request: Any) -> Any:
    raise AssertionError("Dynamic Gym tools must bypass the local tool handler")


class _ModelRequest:
    def __init__(self, context: dict[str, Any]) -> None:
        self.runtime = SimpleNamespace(context=context)
        self.tools = []
        self.tool_choice = None
        self.model_settings = {"existing": "setting"}

    def override(self, **kwargs: Any) -> Any:
        values = {
            "runtime": self.runtime,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "model_settings": self.model_settings,
        }
        values.update(kwargs)
        return SimpleNamespace(**values)


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.timeout: float | None = None
        self.post_args: tuple[str, dict[str, Any], dict[str, str]] | None = None

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        cookies: dict[str, str],
    ) -> httpx.Response:
        self.post_args = (url, json, dict(cookies))
        return httpx.Response(
            200,
            text='{"output":"found"}',
            headers={"set-cookie": "refreshed=cookie"},
            request=httpx.Request("POST", url),
        )
