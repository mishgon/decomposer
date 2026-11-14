import asyncio
import json
from itertools import count
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from tenacity import wait_none

from gyms.wideseek import agentic_rag, agents
from gyms.wideseek.runtime import init_budget

SOLARIS = "https://en.wikipedia.org/wiki/Solaris_(novel)"
CONTEXT = {"agent_run_budget_seconds": 180.0, "agent_shutdown_grace_seconds": 180.0}


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


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
        AIMessage(content="Follow-up report."),
    ])
    monkeypatch.setattr(agentic_rag, "_post", post)
    monkeypatch.setattr(agents, "model", lambda profile: policy)
    episode = tmp_path / "episode"
    init_budget(episode)
    graph = agents.researcher_agentic_rag()
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


def test_researcher_has_no_turn_limit_within_its_recursion_limit(monkeypatch, tmp_path):
    async def post(method, payload):
        return {"results": [{"page_id": 64199, "url": SOLARIS, "title": "Solaris (novel)", "snippet": "Solaris"}]}

    # The fake model returns to its first response after its last, so a spare one keeps i counting calls.
    policy = ToolCallingModel(responses=[calls(("search", {"query": f"Solaris {turn}"})) for turn in range(80)]
                              + [calls(("finish", {"answer": "Lem", "sources": [SOLARIS]})), AIMessage(content="unused")])
    monkeypatch.setattr(agentic_rag, "_post", post)
    monkeypatch.setattr(agents, "model", lambda profile: policy)
    episode = tmp_path / "episode"
    init_budget(episode)
    state = asyncio.run(agents.researcher_agentic_rag().ainvoke(
        {"messages": [HumanMessage("Who wrote Solaris?")]},
        {"configurable": {"thread_id": "researcher"},
         "recursion_limit": agents.RESEARCHER_RECURSION_LIMITS["student_rag"]},
        context={"directory": str(episode), **CONTEXT}))
    assert policy.i == 81
    assert state["messages"][-1].type == "ai" and state["messages"][-1].content == f"Lem\n\nSources:\n{SOLARIS}"


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
    args = run.create_parser().parse_args(["--agent", "student_rag", "--output", str(tmp_path / "run"),
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
        180.0, 180.0)


def test_student_rag_is_student_180_over_agentic_rag_researchers(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    captured = []
    monkeypatch.setattr(agents, "create_decomposer_agent", lambda **kw: captured.append(kw))
    for agent in ("student_180", "student_rag"):
        agents.decomposer("policy", "checkpoint", "http://worker", agent=agent)
    student, rag = (kw["agent_types"][0] for kw in captured)
    assert rag == {**student, "assistant_id": "researcher_agentic_rag"}
    assert [kw["agent_recursion_limit"] for kw in captured] == [410, 10000]
    for table in (agents.AGENT_MODELS, agents.SUBAGENT_MODELS, agents.RUN_BUDGETS):
        assert table["student_rag"] == table["student_180"]
    for gym in ("wideseek", "deepdive"):
        graphs = json.loads((root / f"gyms/{gym}/langgraph.json").read_text())["graphs"]
        assert graphs["researcher_agentic_rag"] == "gyms.wideseek.agents:researcher_agentic_rag"
