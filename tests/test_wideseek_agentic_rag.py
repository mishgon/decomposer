import asyncio
import json
from itertools import count
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from gyms.wideseek import agentic_rag, agents
from gyms.wideseek.runtime import init_budget

SOLARIS = "https://en.wikipedia.org/w/index.php?title=Solaris%20%28novel%29"
LEM = "https://en.wikipedia.org/w/index.php?title=Stanis%C5%82aw%20Lem"
PAGE = ("Solaris (novel)\n\nSolaris is a 1961 [science fiction](https://en.wikipedia.org/w/index.php/science%20fiction) "
        "novel by Polish writer [Stanisław Lem](https://en.wikipedia.org/w/index.php/Stanis%C5%82aw%20Lem). "
        + "It follows a crew of scientists on a space station. " * 6 + "\n\nPlot\n\nKelvin arrives at the station.")


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


CALL_IDS = count()


def calls(*calls):
    return AIMessage(content="", tool_calls=[
        {"id": f"{name}-{next(CALL_IDS)}", "name": name, "args": args} for name, args in calls])


def test_pages_read_as_plain_text_in_lines_of_at_most_200_characters():
    lines = agentic_rag.page_lines({"contents": PAGE})
    assert lines[:2] == ["Solaris (novel)", ""]
    assert lines[2].startswith("Solaris is a 1961 science fiction novel by Polish writer Stanisław Lem. It follows")
    assert all(len(line) <= 200 for line in lines) and len(lines[2]) > 150
    assert lines[-3:] == ["Plot", "", "Kelvin arrives at the station."]
    assert "](" not in "".join(lines)


def test_researcher_resolves_searched_pages_and_ends_with_its_finish_report(monkeypatch, tmp_path):
    requested = []

    async def request_tool(endpoint, payload, runtime):
        requested.append((endpoint, payload))
        if endpoint == "/retrieve":
            return [{"url": SOLARIS, "contents": PAGE}, {"url": SOLARIS, "contents": "A later passage."},
                    {"url": LEM, "contents": "Stanisław Lem was a Polish writer."}]
        return {"url": SOLARIS, "contents": PAGE}

    policy = ToolCallingModel(responses=[
        calls(("search", {"query": "Solaris novel author"})),
        calls(("open", {"url": "https://en.wikipedia.org/w/index.php?title=Kelvin"}),
              ("find", {"url": SOLARIS, "pattern": "LEM"}), ("find", {"url": SOLARIS, "pattern": "("})),
        calls(("open", {"url": SOLARIS, "loc": 2})),
        calls(("finish", {"answer": "Stanisław Lem", "sources": ["https://en.wikipedia.org/w/index.php?title=Kelvin"]})),
        calls(("finish", {"answer": "Stanisław Lem", "sources": [SOLARIS]})),
        AIMessage(content="Follow-up report."),
    ])
    monkeypatch.setattr(agentic_rag, "request_tool", request_tool)
    monkeypatch.setattr(agents, "model", lambda profile: policy)
    episode = tmp_path / "episode"
    init_budget(episode)
    graph = agents.researcher_agentic_rag()
    config = {"configurable": {"thread_id": "researcher"}}
    context = {"directory": str(episode), "agent_run_budget_seconds": 180.0, "agent_shutdown_grace_seconds": 180.0}
    state = asyncio.run(graph.ainvoke({"messages": [HumanMessage("Who wrote Solaris?")]}, config, context=context))

    messages = state["messages"]
    assert messages[-1].type == "ai" and not messages[-1].tool_calls
    assert messages[-1].content == f"Stanisław Lem\n\nSources:\n{SOLARIS}"
    assert policy.i == 5
    assert state["pages"] == [SOLARIS, LEM]
    assert requested == [("/retrieve", {"queries": ["Solaris novel author"], "topk": 10}),
                         ("/access", {"urls": [SOLARIS]}), ("/access", {"urls": [SOLARIS]})]
    search, unknown, found, invalid_pattern, opened, invalid, valid = (
        json.loads(m.content) for m in messages if m.type == "tool")
    assert search == [{"url": SOLARIS, "title": "Solaris (novel)",
                       "snippet": agentic_rag.plain(PAGE)[:300]},
                      {"url": LEM, "title": "Stanisław Lem", "snippet": "Stanisław Lem was a Polish writer."}]
    assert unknown["error"] == "url_not_in_session"
    assert found["total_matches"] == 1 and found["matches"][0]["line"] == 2
    assert found["matches"][0]["excerpt"].startswith("...") and "Stanisław Lem" in found["matches"][0]["excerpt"]
    assert invalid_pattern["error"] == "invalid_pattern"
    lines = agentic_rag.page_lines({"contents": PAGE})
    assert opened == {"lines": "\n".join(lines[2:18]), "total_lines_on_page": len(lines), "loc": 2, "num_lines": 16}
    assert invalid["details"] == [{"field": "sources", "urls": ["https://en.wikipedia.org/w/index.php?title=Kelvin"],
                                   "message": "Use only source URLs returned in this session."}]
    assert valid == {"success": True}
    # A follow-up task on the same thread runs the model again.
    state = asyncio.run(graph.ainvoke({"messages": [*messages, HumanMessage("And when?")]}, config, context=context))
    assert state["messages"][-1].content == "Follow-up report."


@pytest.mark.parametrize("sources, report", [([SOLARIS], f"Lem\n\nSources:\n{SOLARIS}"), (["https://unknown"], "Lem")])
def test_researcher_gets_32_calls_and_the_last_may_only_finish(monkeypatch, tmp_path, sources, report):
    bound = []

    class RecordingModel(ToolCallingModel):
        def bind_tools(self, tools, **kwargs):
            bound.append(sorted(tool.name for tool in tools))
            return self

    async def request_tool(endpoint, payload, runtime):
        return [{"url": SOLARIS, "contents": PAGE}]

    policy = RecordingModel(responses=[calls(("search", {"query": f"Solaris {turn}"})) for turn in range(31)]
                            + [calls(("finish", {"answer": "Lem", "sources": sources}))])
    monkeypatch.setattr(agentic_rag, "request_tool", request_tool)
    monkeypatch.setattr(agents, "model", lambda profile: policy)
    episode = tmp_path / "episode"
    init_budget(episode)
    state = asyncio.run(agents.researcher_agentic_rag().ainvoke(
        {"messages": [HumanMessage("Who wrote Solaris?")]},
        {"configurable": {"thread_id": "researcher"}, "recursion_limit": 410},
        context={"directory": str(episode), "agent_run_budget_seconds": 180.0, "agent_shutdown_grace_seconds": 180.0}))
    assert policy.i == agentic_rag.MAX_TURNS == 32
    assert bound[0] == ["find", "finish", "open", "search"] and bound[-1] == ["finish"]
    assert [m.content for m in state["messages"] if m.type == "human"][-1] == agentic_rag.FINAL_ATTEMPT
    assert state["messages"][-1].type == "ai" and state["messages"][-1].content == report


def test_student_rag_run_records_its_researchers_prompt(monkeypatch, tmp_path):
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
    response = MagicMock()
    response.json.return_value = {"status": "ready"}
    http = MagicMock()
    http.__aenter__ = AsyncMock(return_value=SimpleNamespace(get=AsyncMock(return_value=response)))
    http.__aexit__ = AsyncMock()
    monkeypatch.setattr(run.httpx, "AsyncClient", lambda **kwargs: http)
    harness = asyncio.run(run.describe_run(args))[2]["settings"]["harness"]
    assert harness["agent_system_prompt"] == agentic_rag.SYSTEM_PROMPT
    assert (harness["agent_run_budget_seconds"], harness["agent_shutdown_grace_seconds"]) == (180.0, 180.0)


def test_student_rag_is_student_180_over_agentic_rag_researchers(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    captured = []
    monkeypatch.setattr(agents, "create_decomposer_agent", lambda **kw: captured.append(kw))
    for agent in ("student_180", "student_rag"):
        agents.decomposer("policy", "checkpoint", "http://worker", agent=agent)
    student, rag = (kw["agent_types"][0] for kw in captured)
    assert rag == {**student, "assistant_id": "researcher_agentic_rag"}
    for table in (agents.AGENT_MODELS, agents.SUBAGENT_MODELS, agents.RUN_BUDGETS):
        assert table["student_rag"] == table["student_180"]
    for gym in ("wideseek", "deepdive"):
        graphs = json.loads((root / f"gyms/{gym}/langgraph.json").read_text())["graphs"]
        assert graphs["researcher_agentic_rag"] == "gyms.wideseek.agents:researcher_agentic_rag"
