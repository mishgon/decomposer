import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from langchain_core.messages import AIMessage, message_to_dict
import pytest

from gyms.browsecomp import tools, run, evaluate
from evals.browsecomp.run import summarize


def test_tools_isolate_url_access_and_preserve_state(monkeypatch):
    request = AsyncMock(return_value={"results": [
        {"url": "https://page", "page_id": "p1", "title": "Page", "snippet": "evidence"}]})
    monkeypatch.setattr(tools, "request", request)
    runtime = SimpleNamespace(state={}, tool_call_id="s1")
    update = asyncio.run(tools.search.coroutine("query", runtime)).update
    assert update["pages"] == {"https://page": "p1"}
    assert update["searches"] == 1
    assert asyncio.run(tools.fetch.coroutine("https://page", runtime))["error"] == "url_not_in_session"
    request.assert_awaited_once()
    runtime.state = update
    request.return_value = {"content": "evidence"}
    assert asyncio.run(tools.fetch.coroutine("https://page", runtime)) == {"content": "evidence"}
    request.assert_awaited_with("/fetch", {"page_id": "p1"})
    runtime.state["searches"] = 10
    count = request.await_count
    assert asyncio.run(tools.search.coroutine("query", runtime)).update["searches"] == 0
    assert request.await_count == count


def test_retrieval_contract(monkeypatch):
    calls = []
    def respond(request):
        calls.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/search":
            return httpx.Response(200, json={"results": [{"page_id": "p1"}]})
        return httpx.Response(200, json={"content": "evidence"})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(respond), **kwargs))
    asyncio.run(run.preflight("http://retrieval"))
    assert [c[0] for c in calls] == ["/search", "/fetch"]
    assert all(c[1]["source"] == "browsecomp_plus" for c in calls)
    monkeypatch.setenv("BROWSECOMP_RETRIEVAL_URL", "http://retrieval")
    asyncio.run(tools.request("/fetch", {"page_id": "p1"}))
    assert calls[-1][1] == {"page_id": "p1", "source": "browsecomp_plus"}


def test_judge_errors_are_unscored_and_inclusion_is_not_a_pass(monkeypatch, tmp_path):
    model = SimpleNamespace(bind=lambda **kwargs: model,
                            ainvoke=AsyncMock(return_value=AIMessage(content='{"match": false}')))
    monkeypatch.setattr(evaluate, "create_model", lambda _: model)
    task = {"question": "Which city?", "answer": "New York"}
    output = '```json\n{"final_answer":"York"}\n```'
    result = asyncio.run(evaluate.evaluate(task, output, tmp_path))
    assert result["score"] == 0
    model.ainvoke.assert_awaited_once()
    model.ainvoke.return_value = AIMessage(content="not a verdict")
    assert asyncio.run(evaluate.evaluate(task, output, tmp_path))["score"] is None
    assert "error" in json.loads((tmp_path / "judge.json").read_text())
    assert evaluate.extract_answer("plain text") == ""


@pytest.mark.parametrize("error", [None, TimeoutError("stopped")])
def test_episode_artifacts_and_partial_capture(monkeypatch, tmp_path, error):
    directory = tmp_path / "run" / "traces" / "task-r001"
    directory.mkdir(parents=True)
    (directory / "task.json").write_text(json.dumps({"task_id": "task", "question": "Q", "answer": "A"}))
    @asynccontextmanager
    async def server(*args, **kwargs):
        yield "http://fake"
    capture = AsyncMock(return_value=({"messages": [message_to_dict(AIMessage(
        content='```json\n{"final_answer":"A"}\n```'))]}, error))
    monkeypatch.setattr(run, "agent_server", server)
    monkeypatch.setattr(run, "invoke_and_capture", capture)
    args = run.parser().parse_args(["--episode", "--output-dir", str(directory), "--agent", "researcher"])
    asyncio.run(run.episode(args))
    result = json.loads((directory / "result.json").read_text())
    assert result["status"] == ("timeout" if error else "finished")
    assert result["evaluation"]["score"] == 1
    assert (directory / "trace.json").exists()
    assert (directory / "usage.json").exists()
    assert capture.call_args.kwargs["config"] == {"recursion_limit": 410}


def test_pass_metrics_count_infra_failures():
    rows = [{"task": "a", "status": "finished", "evaluation": {"score": 1, "passed": True}},
            {"task": "a", "status": "timeout", "evaluation": {"score": 1, "passed": True}},
            {"task": "a", "status": "error", "evaluation": {"score": None, "passed": False}}]
    summary = summarize({"episodes": rows, "tasks": ["a", "b"], "repetitions": 3, "status": "completed"})
    assert summary["pass@1"] == pytest.approx(1 / 6)
    assert summary["pass@3"] == .5
    assert summary["pass^3"] == 0
    assert summary["unscored"] == 1


def test_scheduler_records_crashed_episodes(monkeypatch, tmp_path):
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"task_id": "a", "question": "Q", "answer": "A"}) + "\n")
    monkeypatch.setattr(run, "preflight", AsyncMock())
    async def spawn(*command, **kwargs):
        return SimpleNamespace(returncode=1, wait=AsyncMock(return_value=1))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    args = run.parser().parse_args(["--data", str(data), "--all", "-n", "3", "--output-dir", str(tmp_path / "runs")])
    root = asyncio.run(run.main(args))
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["episodes"]) == 3
    assert all(row["status"] == "error" for row in manifest["episodes"])
    assert summarize(manifest)["pass@3"] == 0


def test_failed_preflight_does_not_schedule(monkeypatch, tmp_path):
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"task_id": "a", "question": "Q", "answer": "A"}) + "\n")
    monkeypatch.setattr(run, "preflight", AsyncMock(side_effect=ConnectionError("retrieval down")))
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    args = run.parser().parse_args(["--data", str(data), "--all", "--output-dir", str(tmp_path / "runs")])
    with pytest.raises(ConnectionError):
        asyncio.run(run.main(args))
    spawn.assert_not_called()
    assert not (tmp_path / "runs").exists()


def test_compiled_researcher_search_fetch_and_fresh_thread(monkeypatch, tmp_path):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from gyms.browsecomp import agents

    class Model(FakeMessagesListChatModel):
        model_name: str = "test"

        def bind_tools(self, tools, **kwargs):
            return self

    model = Model(responses=[
        AIMessage(content="", tool_calls=[{"id": "s", "name": "search", "args": {"query": "Q"}}]),
        AIMessage(content="", tool_calls=[{"id": "f", "name": "fetch", "args": {"url": "https://page"}}]),
        AIMessage(content='```json\n{"final_answer":"A"}\n```'),
    ])
    monkeypatch.setattr(agents, "create_model", lambda _: model)
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "calls.jsonl"))
    request = AsyncMock(side_effect=[{"results": [
        {"url": "https://page", "page_id": "p1", "title": "Page", "snippet": "Evidence"}]},
        {"content": "A"}])
    monkeypatch.setattr(tools, "request", request)
    graph = agents.researcher()
    state = asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "Q"}]}))
    assert state["pages"] == {"https://page": "p1"}
    assert state["searches"] == 1
    assert evaluate.extract_answer(state["messages"][-1].content) == "A"
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert sum(r["status"] == "success" for r in records) == 3
    # A separate invocation must not inherit URLs from the preceding episode.
    model.responses = [AIMessage(content="", tool_calls=[{"id": "f2", "name": "fetch", "args": {"url": "https://page"}}]),
                       AIMessage(content="No evidence")]
    model.i = 0
    fresh = asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "Q2"}]}))
    assert "url_not_in_session" in fresh["messages"][-2].content
    assert request.await_count == 2


def test_real_agent_server_with_mock_retrieval(monkeypatch, tmp_path):
    """Exercise HTTP transport, tools, checkpoints and capture without hosted models."""
    import importlib.util
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    if importlib.util.find_spec("langgraph_cli") is None:
        pytest.skip("LangGraph CLI is not installed")

    class Retrieval(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["source"] == "browsecomp_plus"
            result = {"results": [{"url": "https://page", "page_id": "p1", "title": "Page", "snippet": "A"}]} if self.path == "/search" else {"content": "A"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    service = ThreadingHTTPServer(("127.0.0.1", 0), Retrieval)
    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()
    fixture = tmp_path / "graphs.py"
    fixture.write_text('''
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from gyms.browsecomp import agents
class Model(FakeMessagesListChatModel):
    model_name: str = "test"
    def bind_tools(self, tools, **kwargs):
        return self
def researcher():
    agents.create_model = lambda _: Model(responses=[
        AIMessage(content="", tool_calls=[{"id":"s", "name":"search", "args":{"query":"Q"}}]),
        AIMessage(content="", tool_calls=[{"id":"f", "name":"fetch", "args":{"url":"https://page"}}]),
        AIMessage(content='```json\\n{"final_answer":"A"}\\n```'),
    ])
    return agents.researcher()
''')
    config = tmp_path / "langgraph.json"
    config.write_text(json.dumps({"dependencies": [str(run.REPO_ROOT)],
                                 "graphs": {"researcher": str(fixture) + ":researcher"}}))
    monkeypatch.setenv("PYTHONPATH", f"{run.REPO_ROOT / 'src'}:{run.REPO_ROOT}")
    monkeypatch.setenv("BROWSECOMP_RETRIEVAL_URL", f"http://127.0.0.1:{service.server_port}")
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    async def smoke():
        async with run.agent_server(config, port=port) as url:
            return await run.invoke_and_capture(url, "researcher",
                {"messages": [{"role": "user", "content": "Q"}]}, timeout=30,
                config={"recursion_limit": 410})
    try:
        state, error = asyncio.run(smoke())
        assert error is None
        assert state["pages"] == {"https://page": "p1"}
        assert state["searches"] == 1
        assert state["messages"][-1]["content"].endswith("```")
        assert (tmp_path / "model_calls.jsonl").exists()
    finally:
        service.shutdown()
        service.server_close()
        thread.join()
