"""Check gym worker execution with seeded resource-server sessions."""

import asyncio
import importlib
import json

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from responses_api_agents.decomposer_agent.subagents import graph as nemo_graph


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        assert tools[0]["function"]["name"] == "lookup"
        return self


@pytest.mark.parametrize("gym", ["tau2_gym", "workplace_assistant"])
def test_worker_routes_native_tools_with_session(gym, monkeypatch):
    graph = importlib.import_module(f"gyms.{gym}.subagents.graph")
    context = {
        "body": {
            "tools": [{
                "type": "function",
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }],
        },
        "resource_server_url": "http://resources",
        "resource_server_cookies": {"session": "seeded"},
    }
    requests = []

    def resource_server(request):
        requests.append(request)
        assert str(request.url) == "http://resources/lookup"
        assert request.headers["cookie"] == "session=seeded"
        assert json.loads(request.content) == {"query": "task"}
        return httpx.Response(200, json={"answer": "found"}, headers={"set-cookie": "session=updated"})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(nemo_graph, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(resource_server), **kwargs
    ))
    model = ToolCallingModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "lookup", "args": {"query": "task"}, "id": "lookup_1", "type": "tool_call",
        }]),
        AIMessage(content="found"),
    ])
    state = asyncio.run(graph._create_subagent(model).ainvoke(
        {"messages": [HumanMessage(content="Look up task")]}, context=context
    ))
    assert len(requests) == 1
    tool_message = next(message for message in state["messages"] if isinstance(message, ToolMessage))
    assert tool_message.status == "success"
    assert json.loads(tool_message.content) == {"answer": "found"}
    assert state["messages"][-1].content == "found"
    assert context["resource_server_cookies"] == {"session": "updated"}
