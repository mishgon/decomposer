import asyncio
import json
from itertools import count

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from tenacity import wait_none

from gyms.wideseek import agentic_rag, agents
from gyms.wideseek.runtime import init_budget

SOLARIS = "https://en.wikipedia.org/wiki/Solaris_(novel)"
CONTEXT = {}


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture(autouse=True)
def token_count(monkeypatch):
    monkeypatch.setattr(agentic_rag, "prompt_tokens", lambda messages, tools: 100)


CALL_IDS = count()


def calls(*calls):
    return AIMessage(content="", tool_calls=[
        {"id": f"{name}-{next(CALL_IDS)}", "name": name, "args": args} for name, args in calls])


def test_researcher_resolves_searched_pages_and_ends_with_its_finish_report(monkeypatch, tmp_path):
    posted = []

    async def post(method, payload):
        posted.append((method, payload))
        return {"search": {"results": [{"page_id": 64199, "url": SOLARIS, "title": "Solaris (novel)",
                                        "snippet": "Solaris () is a 1961 science fiction novel"}]},
                "find": {"matches": [{"line": 0, "excerpt": "... Polish writer Stanisław Lem ..."}],
                         "total_matches": 1, "total_lines_on_page": 124}}[method]

    policy = ToolCallingModel(responses=[
        calls(("search", {"query": "Solaris novel author"})),
        calls(("open", {"url": "https://en.wikipedia.org/wiki/Lem"}), ("find", {"url": SOLARIS + "/", "pattern": "Lem"})),
        calls(("finish", {"answer": "Stanisław Lem", "sources": ["https://en.wikipedia.org/wiki/Lem"]})),
        calls(("finish", {"answer": "Stanisław Lem", "sources": [SOLARIS + "/"]})),
        calls(("finish", {"answer": "Follow-up report.", "sources": []})),
    ])
    monkeypatch.setattr(agentic_rag, "_post", post)
    monkeypatch.setattr(agents, "model", lambda profile: policy)
    episode = tmp_path / "episode"
    init_budget(episode)
    graph = agents.researcher()
    config = {"configurable": {"thread_id": "researcher"}}
    context = {"directory": str(episode), **CONTEXT}
    state = asyncio.run(graph.ainvoke({"messages": [HumanMessage("Who wrote Solaris?")]}, config, context=context))

    messages = state["messages"]
    assert messages[-1].type == "ai" and not messages[-1].tool_calls
    assert messages[-1].content == f"Stanisław Lem\n\nSources:\n{SOLARIS}"
    assert policy.i == 4
    assert state["pages"] == {SOLARIS: 64199}
    assert posted == [("search", {"query": "Solaris novel author", "limit": 5}),
                      ("find", {"page_id": 64199, "pattern": "Lem", "max_entries": 16})]
    search, unknown, found, invalid, valid = (json.loads(m.content) for m in messages if m.type == "tool")
    assert search == [{"url": SOLARIS, "title": "Solaris (novel)",
                       "snippet": "Solaris () is a 1961 science fiction novel"}]
    assert unknown["error"] == "url_not_in_session"
    assert found["total_matches"] == 1
    assert invalid["details"] == [{"field": "sources", "urls": ["https://en.wikipedia.org/wiki/Lem"],
                                   "message": "Use only source URLs returned in this session."}]
    assert valid == {"success": True}
    assert len(list((episode / "tool_calls").glob("*.json"))) == 2
    # A follow-up task on the same thread runs the model again.
    state = asyncio.run(graph.ainvoke({"messages": [*messages, HumanMessage("And when?")]}, config, context=context))
    assert state["messages"][-1].content == "Follow-up report."


def test_search_api_waits_out_transient_errors_and_stops_on_others(monkeypatch):
    requests, statuses = [], [503, 200, 401]

    def respond(request):
        requests.append(request)
        return httpx.Response(statuses[len(requests) - 1], json={"results": []}, request=request)

    client = httpx.AsyncClient
    monkeypatch.setattr(agentic_rag.httpx, "AsyncClient",
                        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.delenv("AGENTIC_RAG_URL", raising=False)
    monkeypatch.setenv("AGENTIC_RAG_API_KEY", "test-key")
    post = agentic_rag._post.retry_with(wait=wait_none())
    assert asyncio.run(post("search", {"query": "q", "limit": 5})) == {"results": []}
    assert [str(r.url) for r in requests] == ["https://lmrouter.2a2i.org/rag/search"] * 2
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content) == {"query": "q", "limit": 5, "source": "wikipedia"}
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(post("search", {"query": "q", "limit": 5}))
    assert len(requests) == 3


def test_student_rag_run_records_the_search_api_and_its_researchers_prompt(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from gyms.wideseek import run

    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"task_id": "task", "question": "Q"}) + "\n")
    args = run.create_parser().parse_args(["--agent", "student", "--output", str(tmp_path / "run"),
                                           "--data", str(data)])
    args.mode = "decomposer"
    monkeypatch.setattr(run.subprocess, "check_output", lambda *a, **kw: "commit")
    monkeypatch.setattr(run, "model", lambda _: SimpleNamespace(model_name="test"))
    monkeypatch.delenv("AGENTIC_RAG_URL", raising=False)
    monkeypatch.setenv("AGENTIC_RAG_API_KEY", "test-key")
    response = MagicMock()
    response.json.return_value = {"status": "ok"}
    get = AsyncMock(return_value=response)
    http = MagicMock()
    http.__aenter__ = AsyncMock(return_value=SimpleNamespace(get=get))
    http.__aexit__ = AsyncMock()
    monkeypatch.setattr(run.httpx, "AsyncClient", lambda **kwargs: http)
    settings = asyncio.run(run.describe_run(args))[2]["settings"]
    assert get.await_args_list[0].args == ("https://lmrouter.2a2i.org/rag/health",)
    assert get.await_args_list[0].kwargs == {"headers": {"Authorization": "Bearer test-key"}}
    assert settings["retrieval"] == {"search_api": "https://lmrouter.2a2i.org/rag", "source": "wikipedia",
                                     "health": {"status": "ok"}}
    assert settings["harness"]["agent_system_prompt"] == agentic_rag.SYSTEM_PROMPT
    assert (settings["harness"]["agent_run_budget_seconds"], settings["harness"]["agent_shutdown_grace_seconds"]) == (
        None, None)



@pytest.mark.parametrize("responses,status", [
    ([AIMessage(content="Unsubmitted answer") for _ in range(3)], "invalid_tool_call"),
    ([calls(("search", {"query": "same"})) for _ in range(3)], "loop_detected"),
    ([calls(("search", {"query": 123})) for _ in range(3)], "invalid_tool_call"),
])
def test_protocol_rejects_plain_answers_repeats_and_invalid_arguments(monkeypatch, tmp_path, responses, status):
    from unittest.mock import AsyncMock
    post = AsyncMock(return_value={"results": []})
    monkeypatch.setattr(agentic_rag, "_post", post)
    tmp_path = tmp_path / "episode"
    init_budget(tmp_path)
    policy = ToolCallingModel(responses=responses + [AIMessage(content="unused")])
    state = asyncio.run(agents.react(policy, None).ainvoke(
        {"messages": [HumanMessage("Question")]}, {"configurable": {"thread_id": "test"}}, context={"directory": str(tmp_path), **CONTEXT}))
    assert state["end_status"] == status
    assert state["messages"][-1].content == ""
    assert policy.i == 3
    assert post.await_count == (2 if status == "loop_detected" else 0)


def test_tools_share_sequential_state_and_stop_immediately_after_finish(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    post = AsyncMock(return_value={"results": [{"page_id": 1, "url": SOLARIS, "title": "Solaris", "snippet": "Lem"}]})
    monkeypatch.setattr(agentic_rag, "_post", post)
    tmp_path = tmp_path / "episode"
    init_budget(tmp_path)
    policy = ToolCallingModel(responses=[calls(
        ("search", {"query": "Solaris"}),
        ("finish", {"answer": "Lem", "sources": [SOLARIS]}),
        ("search", {"query": "must not run"}))])
    state = asyncio.run(agents.react(policy, None).ainvoke(
        {"messages": [HumanMessage("Question")]}, {"configurable": {"thread_id": "test"}}, context={"directory": str(tmp_path), **CONTEXT}))
    assert state["end_status"] == "finished"
    assert post.await_count == 1
    assert state["messages"][-1].content == f"Lem\n\nSources:\n{SOLARIS}"
    tool_messages = [m for m in state["messages"] if m.type == "tool"]
    assert len(tool_messages) == 3
    assert json.loads(tool_messages[-1].content)["error"] == "session_ended"


def test_grace_keeps_finish_available(monkeypatch, tmp_path):
    import time
    from langchain.agents.middleware import AgentMiddleware

    class ExpireWork(AgentMiddleware):
        def before_agent(self, state, runtime):
            return {"shutdown_at": time.monotonic() - 1, "timeout_at": time.monotonic() + 180}

    class InspectTools(AgentMiddleware):
        async def awrap_model_call(self, request, handler):
            assert request.tools == agentic_rag.TOOL_SCHEMAS
            assert request.model_settings["max_tokens"] == 4096
            assert request.messages[-1].content == agentic_rag.FINISH_REQUEST
            return await handler(request)

    tmp_path = tmp_path / "episode"
    init_budget(tmp_path, tokens=100000)
    policy = ToolCallingModel(responses=[calls(("finish", {"answer": "Unknown", "sources": []}))])
    state = asyncio.run(agents.react(policy, None, middleware=[ExpireWork(), InspectTools()]).ainvoke(
        {"messages": [HumanMessage("Question")]}, {"configurable": {"thread_id": "test"}}, context={"directory": str(tmp_path), **CONTEXT}))
    assert state["end_status"] == "finished"
    assert state["messages"][-1].content == "Unknown"


def test_context_limit_stops_before_model_call(monkeypatch, tmp_path):
    monkeypatch.setattr(agentic_rag, "prompt_tokens", lambda messages, tools: 65536)
    tmp_path = tmp_path / "episode"
    init_budget(tmp_path)
    policy = ToolCallingModel(responses=[AIMessage(content="must not run"), AIMessage(content="unused")])
    state = asyncio.run(agents.react(policy, None).ainvoke(
        {"messages": [HumanMessage("Question")]}, {"configurable": {"thread_id": "test"}}, context={"directory": str(tmp_path), **CONTEXT}))
    assert state["end_status"] == "context_limit"
    assert policy.i == 0


@pytest.mark.parametrize("last_call, expected_status", [
    (("finish", {"answer": "Partial answer", "sources": []}), "finished"),
    (("search", {"query": "must not execute"}), "turn_limit"),
    (("finish", {"answer": "", "sources": []}), "turn_limit"),
])
def test_last_of_32_model_calls_only_allows_finish_and_resets_next_run(
    monkeypatch, tmp_path, last_call, expected_status,
):
    from unittest.mock import AsyncMock
    from langgraph.checkpoint.memory import InMemorySaver

    post = AsyncMock(return_value={"results": []})
    monkeypatch.setattr(agentic_rag, "_post", post)
    tmp_path = tmp_path / "episode"
    init_budget(tmp_path)
    policy = ToolCallingModel(responses=[
        *(calls(("search", {"query": f"question {i}"})) for i in range(31)),
        calls(last_call),
        calls(("finish", {"answer": "Next task", "sources": []})),
    ])
    graph = agents.react(policy, InMemorySaver())
    config = {"configurable": {"thread_id": "test"}, "recursion_limit": 500}
    context = {"directory": str(tmp_path), **CONTEXT}
    state = asyncio.run(graph.ainvoke({"messages": [HumanMessage("Question")]}, config, context=context))
    assert policy.i == 32
    assert post.await_count == 31
    assert state["completed_turns"] == 32
    assert state["shutdown_at"] is None
    assert state["timeout_at"] is None
    assert state["end_status"] == expected_status
    assert sum(m.content == agentic_rag.LAST_TURN_REQUEST for m in state["messages"]) == 1
    if last_call[0] == "search":
        tool_messages = [m for m in state["messages"] if m.type == "tool"]
        assert json.loads(tool_messages[-1].content)["error"] == "finish_required"
    state = asyncio.run(graph.ainvoke({"messages": [HumanMessage("Another task")]}, config, context=context))
    assert state["completed_turns"] == 1
    assert state["end_status"] == "finished"
    assert state["messages"][-1].content == "Next task"


def test_output_limit_stops_before_another_model_call(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    monkeypatch.setattr(agentic_rag, "_post", AsyncMock(return_value={"results": []}))
    tmp_path = tmp_path / "episode"
    init_budget(tmp_path)
    response = calls(("search", {"query": "question"}))
    response.usage_metadata = {"input_tokens": 100, "output_tokens": 65536, "total_tokens": 65636}
    policy = ToolCallingModel(responses=[response, AIMessage(content="must not run")])
    state = asyncio.run(agents.react(policy, None).ainvoke(
        {"messages": [HumanMessage("Question")]}, {"configurable": {"thread_id": "test"}},
        context={"directory": str(tmp_path), **CONTEXT}))
    assert state["end_status"] == "output_limit"
    assert policy.i == 1


def test_tool_execution_cannot_overrun_shutdown_grace(monkeypatch, tmp_path):
    async def post(method, payload):
        await asyncio.sleep(10)

    monkeypatch.setattr(agentic_rag, "_post", post)
    tmp_path = tmp_path / "episode"
    init_budget(tmp_path)
    policy = ToolCallingModel(responses=[calls(("search", {"query": "question"}))])
    with pytest.raises(TimeoutError):
        asyncio.run(agents.react(policy, None).ainvoke(
            {"messages": [HumanMessage("Question")]}, {"configurable": {"thread_id": "test"}},
            context={"directory": str(tmp_path), "agent_run_budget_seconds": .1,
                     "agent_shutdown_grace_seconds": .1}))


def test_registered_tool_middleware_runs_in_order_and_can_reject(monkeypatch, tmp_path):
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage
    from unittest.mock import AsyncMock
    events = []

    class Outer(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            events.append(("before", request.tool_call["name"]))
            result = await handler(request)
            events.append(("after", request.tool_call["name"]))
            return result

    class RejectSearch(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            events.append(("inner", request.tool_call["name"]))
            if request.tool_call["name"] == "search":
                return ToolMessage('blocked', name='search', tool_call_id=request.tool_call['id'])
            return await handler(request)

    post = AsyncMock()
    monkeypatch.setattr(agentic_rag, "_post", post)
    tmp_path = tmp_path / "episode"
    init_budget(tmp_path)
    policy = ToolCallingModel(responses=[calls(("search", {"query": "q"})),
                                        calls(("finish", {"answer": "Unknown", "sources": []}))])
    state = asyncio.run(agents.react(policy, None, middleware=[Outer(), RejectSearch()]).ainvoke(
        {"messages": [HumanMessage("Question")]}, {"configurable": {"thread_id": "test"}},
        context={"directory": str(tmp_path), **CONTEXT}))
    assert state['end_status'] == 'finished'
    assert post.await_count == 0
    assert events == [(stage, name) for name in ('search', 'finish') for stage in ('before', 'inner', 'after')]


def test_wire_schemas_and_followup_history_match_protocol(monkeypatch, tmp_path):
    from pathlib import Path
    from decomposer import models
    from gyms.wideseek.runtime import close_model
    from langgraph.checkpoint.memory import InMemorySaver

    expected = json.loads((Path(__file__).parent / 'fixtures/agentic_rag/tools.json').read_text())
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={
            'id': f'response-{len(requests)}', 'object': 'chat.completion', 'created': 1, 'model': 'occ-4b',
            'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': f'finish-{len(requests)}', 'type': 'function',
                    'function': {'name': 'finish', 'arguments': '{"answer":"Known answer","sources":[]}'}}]}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}})

    monkeypatch.setattr(models.httpx, 'HTTPTransport', lambda **kwargs: httpx.MockTransport(respond))
    monkeypatch.setattr(models.httpx, 'AsyncHTTPTransport', lambda **kwargs: httpx.MockTransport(respond))
    tmp_path = tmp_path / 'episode'
    init_budget(tmp_path)

    async def run():
        policy = models.create_model('lmrouter/occ_4b_non_thinking')
        try:
            graph = agents.react(policy, InMemorySaver())
            for question in ('Question', 'Follow-up'):
                state = await graph.ainvoke({'messages': [HumanMessage(question)]},
                    {'configurable': {'thread_id': 'test'}}, context={'directory': str(tmp_path), **CONTEXT})
                assert state['messages'][-1].content == 'Known answer'
        finally:
            await close_model(policy)

    asyncio.run(run())
    assert len(requests) == 2
    for payload in requests:
        assert payload['tools'] == expected
        assert json.dumps(payload['tools']) == json.dumps(expected)
        assert payload['model'] == 'occ-4b'
        assert payload['chat_template_kwargs']['enable_thinking'] is False
        assert (payload['temperature'], payload['top_p'], payload['top_k']) == (.7, .8, 20)
        assert payload.get('max_tokens', payload.get('max_completion_tokens')) == 4096
        assert payload['messages'][0] == {'role': 'system', 'content': (
            Path(__file__).parent / 'fixtures/agentic_rag/system_prompt.txt').read_text()}
    messages = requests[-1]['messages']
    assert [message['role'] for message in messages] == ['system', 'user', 'assistant', 'tool', 'user']
    assert messages[-2] == {'role': 'tool', 'content': '{"success": true}', 'tool_call_id': 'finish-1'}
    assert not any(message.get('content') == 'Known answer' for message in messages)


def test_shutdown_rejects_search_without_hiding_its_schema(monkeypatch, tmp_path):
    import time
    from langchain.agents.middleware import AgentMiddleware
    from unittest.mock import AsyncMock

    class ExpireWork(AgentMiddleware):
        def before_agent(self, state, runtime):
            return {'shutdown_at': time.monotonic() - 1, 'timeout_at': time.monotonic() + 180}

    post = AsyncMock()
    monkeypatch.setattr(agentic_rag, '_post', post)
    tmp_path = tmp_path / 'episode'
    init_budget(tmp_path)
    policy = ToolCallingModel(responses=[calls(('search', {'query': 'q'})),
        calls(('finish', {'answer': 'Unknown', 'sources': []}))])
    state = asyncio.run(agents.react(policy, None, middleware=[ExpireWork()]).ainvoke(
        {'messages': [HumanMessage('Question')]}, {'configurable': {'thread_id': 'test'}},
        context={'directory': str(tmp_path), **CONTEXT}))
    assert state['end_status'] == 'finished'
    assert post.await_count == 0
    tools = [m for m in state['messages'] if m.type == 'tool']
    assert json.loads(tools[0].content) == {
        'error': 'finish_required', 'message': 'Only finish is allowed on the final attempt.'}


def test_malformed_call_preceding_finish_receives_feedback_in_original_order(monkeypatch, tmp_path):
    tmp_path = tmp_path / 'episode'
    init_budget(tmp_path)
    response = calls(('finish', {'answer': 'Unknown', 'sources': []}))
    finish_id = response.tool_calls[0]['id']
    response.invalid_tool_calls = [{'name': 'search', 'args': '{', 'id': 'malformed',
                                    'error': 'invalid JSON', 'type': 'invalid_tool_call'}]
    response.additional_kwargs['tool_calls'] = [
        {'id': 'malformed', 'type': 'function', 'function': {'name': 'search', 'arguments': '{'}},
        {'id': finish_id, 'type': 'function', 'function': {
            'name': 'finish', 'arguments': '{"answer":"Unknown","sources":[]}'}}]
    policy = ToolCallingModel(responses=[response])
    state = asyncio.run(agents.react(policy, None).ainvoke(
        {'messages': [HumanMessage('Question')]}, {'configurable': {'thread_id': 'test'}},
        context={'directory': str(tmp_path), **CONTEXT}))
    tools = [m for m in state['messages'] if m.type == 'tool']
    assert [m.tool_call_id for m in tools] == ['malformed', finish_id]
    assert json.loads(tools[0].content) == {'error': 'invalid_tool_call_format', 'details': [
        {'path': [], 'message': 'Arguments must be a JSON object.'}]}
    assert state['end_status'] == 'finished'
