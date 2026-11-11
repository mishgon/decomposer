import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
    assert asyncio.run(tools.fetch.coroutine("https://page", runtime))["error"] == "url_not_in_session"
    request.assert_awaited_once()
    runtime.state = update
    request.return_value = {"content": "evidence"}
    assert asyncio.run(tools.fetch.coroutine("https://page", runtime)) == {"content": "evidence"}
    request.assert_awaited_with("/fetch", {"page_id": "p1"})


def test_search_continues_beyond_ten_calls(monkeypatch):
    request = AsyncMock(return_value={"results": [
        {"url": "https://page", "page_id": "p1", "title": "Page", "snippet": "evidence"}]})
    monkeypatch.setattr(tools, "request", request)
    runtime = SimpleNamespace(state={}, tool_call_id="search")
    async def search_repeatedly():
        for i in range(12):
            update = (await tools.search.coroutine(f"query {i}", runtime)).update
            runtime.state.update(update)
            assert json.loads(update["messages"][0].content)["results"][0]["url"] == "https://page"
    asyncio.run(search_repeatedly())
    assert request.await_count == 12


def test_retrieval_contract(monkeypatch):
    calls = []
    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"corpus_revision": run.CORPUS_REVISION})
        if request.url.path == "/retrieve":
            return httpx.Response(200, json={"result": [[{"url": "https://page", "contents": "Evidence"}]]})
        return httpx.Response(200, json={"result": [{"contents": "Evidence"}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(respond), **kwargs))
    asyncio.run(run.preflight("http://retrieval"))
    assert calls == ["/health", "/retrieve", "/access"]


def test_judge_errors_are_unscored_and_inclusion_is_not_a_pass(monkeypatch, tmp_path):
    model = SimpleNamespace(bind=lambda **kwargs: model,
                            ainvoke=AsyncMock(return_value=AIMessage(content='{"match": false}')))
    monkeypatch.setattr(evaluate, "create_model", lambda _: model)
    task = {"question": "Which city?", "answer": "New York"}
    output = 'York'
    result = asyncio.run(evaluate.evaluate(task, output, tmp_path))
    assert result["score"] == 0
    model.ainvoke.assert_awaited_once()
    model.ainvoke.return_value = AIMessage(content="not a verdict")
    assert asyncio.run(evaluate.evaluate(task, output, tmp_path))["score"] is None
    assert "error" in json.loads((tmp_path / "judge.json").read_text())


@pytest.mark.parametrize("agent", list(run.AGENTS))
@pytest.mark.parametrize("error", [None, TimeoutError("stopped")])
def test_episode_artifacts_and_partial_capture(monkeypatch, tmp_path, error, agent):
    directory = tmp_path / "run" / "traces" / "task-r001"
    directory.mkdir(parents=True)
    (directory / "task.json").write_text(json.dumps({"task_id": "task", "question": "Q", "answer": "A"}))
    @asynccontextmanager
    async def server(*args, **kwargs):
        yield "http://fake"
    capture = AsyncMock(return_value=({"messages": [message_to_dict(AIMessage(
        content='A'))]}, error))
    monkeypatch.setattr(run, "agent_server", server)
    monkeypatch.setattr(run, "invoke_and_capture", capture)
    args = run.parser().parse_args(["--episode", "--output-dir", str(directory), "--agent", agent])
    asyncio.run(run.episode(args))
    result = json.loads((directory / "result.json").read_text())
    assert result["status"] == ("timeout" if error else "finished")
    assert result["evaluation"]["score"] == 1
    trace = json.loads((directory / "trace.json").read_text())
    assert (trace["model"], trace["harness"]) == run.AGENTS[agent]
    usage = json.loads((directory / "usage.json").read_text())
    assert ("react" in usage) == (run.AGENTS[agent][1] == "react")
    assert (directory / "usage.json").exists()
    assert capture.call_args.kwargs["config"] == {"recursion_limit": 410}
    assert capture.call_args.kwargs["context"] == {"directory": str(directory / "runtime")}
    assert capture.call_args.args[2]["messages"][0]["content"] == (
        "Q\n\nIf you determine that no further external knowledge is required, "
        "you have to wrap your final answer in \\boxed{}.")


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


def test_compiled_researcher_uses_wideseek_tools(monkeypatch, tmp_path):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from gyms.browsecomp import agents
    from gyms.wideseek import worker
    from gyms.wideseek.runtime import init_budget

    class Model(FakeMessagesListChatModel):
        model_name: str = "test"
        def bind_tools(self, tools, **kwargs):
            return self

    model = Model(responses=[
        AIMessage(content="", tool_calls=[{"id": "s", "name": "search", "args": {"query": "Q"}}]),
        AIMessage(content="", tool_calls=[{"id": "f", "name": "access", "args": {"url": "https://page"}}]),
        AIMessage(content="A"),
    ])
    monkeypatch.setattr(agents, "create_model", lambda _: model)
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "calls.jsonl"))
    request = AsyncMock(side_effect=[[{"url": "https://page", "contents": "Evidence"}], {"contents": "A"}])
    monkeypatch.setattr(worker, "request_tool", request)
    directory = tmp_path / "runtime"
    init_budget(directory)
    state = asyncio.run(agents.researcher().ainvoke({"messages": [{"role": "user", "content": "Q"}]},
        config={"configurable": {"thread_id": "test"}}, context={"directory": str(directory)}))
    assert state["messages"][-1].content == "A"
    assert [call.args[0] for call in request.call_args_list] == ["/retrieve", "/access"]
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().split("\n") if line]
    assert sum(r["status"] == "success" for r in records) == 3


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
            assert "source" not in payload
            result = {"result": [[{"url": "https://page", "contents": "A"}]]} if self.path == "/retrieve" else {"result": [{"contents": "A"}]}
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
        AIMessage(content="", tool_calls=[{"id":"f", "name":"access", "args":{"url":"https://page"}}]),
        AIMessage(content='A'),
    ])
    return agents.researcher()
''')
    config = tmp_path / "langgraph.json"
    config.write_text(json.dumps({"dependencies": [str(run.REPO_ROOT)],
                                 "graphs": {"researcher": str(fixture) + ":researcher"}}))
    monkeypatch.setenv("PYTHONPATH", f"{run.REPO_ROOT / 'src'}:{run.REPO_ROOT}")
    monkeypatch.setenv("WS_SEARCH_URL", f"http://127.0.0.1:{service.server_port}")
    run.init_budget(tmp_path / "runtime")
    monkeypatch.setenv("BROWSECOMP_MODEL_LOG", str(tmp_path / "model_calls.jsonl"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    async def smoke():
        async with run.agent_server(config, port=port) as url:
            return await run.invoke_and_capture(url, "researcher",
                {"messages": [{"role": "user", "content": "Q"}]}, timeout=30,
                config={"recursion_limit": 410}, context={"directory": str(tmp_path / "runtime")})
    try:
        state, error = asyncio.run(smoke())
        assert error is None
        assert state["messages"][-1]["content"] == "A"
        assert (tmp_path / "model_calls.jsonl").exists()
    finally:
        service.shutdown()
        service.server_close()
        thread.join()


@pytest.mark.parametrize("name", list(run.AGENTS))
def test_registered_factories_match_trace_models_and_worker_identity(monkeypatch, name):
    from gyms.browsecomp import agents
    monkeypatch.setenv("BROWSECOMP_AGENT_URL", "http://worker")
    monkeypatch.setattr(agents, "create_model", lambda model: model)
    from gyms.wideseek import agents as wideseek_agents
    from gyms.wideseek.worker import search, access
    monkeypatch.setattr(wideseek_agents, "create_agent", lambda model, **kw: {"model": model, **kw})
    monkeypatch.setattr(agents, "create_decomposer_agent", lambda **kw: kw)
    graphs = json.loads(run.Path(agents.__file__).with_name("langgraph.json").read_text())["graphs"]
    assert graphs[name] == f"./agents.py:{name}"
    config = getattr(agents, name)()
    model, harness = run.AGENTS[name]
    if harness == "react":
        assert config["model"] == model
        assert config["tools"] == [search, access]
    else:
        assert config["decomposer_model"] == model
        assert config["agent_run_budget_seconds"] == 180.0
        assert config["agent_shutdown_grace_seconds"] == 180.0
        assert config["agent_types"] == [{
            "agent_type_id": "qwen_3_5_4b_thinking", "assistant_id": "researcher",
            "description": "Qwen3.5-4B thinking agent equipped with search and access tools.",
            "url": "http://worker",
        }]


@pytest.mark.parametrize("mismatch", [None, "agent", "timeout", "running"])
def test_resume_keeps_completed_episodes_and_archives_partial_logs(monkeypatch, tmp_path, mismatch):
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"task_id": "a", "question": "Q", "answer": "A"}) + "\n")
    root = tmp_path / "run"
    complete = root / "traces/a-r001"
    complete.mkdir(parents=True)
    (complete / "trace.json").write_text('original trace')
    partial = root / "traces/a-r002"
    partial.mkdir()
    (partial / "model_calls.jsonl").write_text('interrupted request')
    args = run.parser().parse_args(["--data", str(data), "--all", "-n", "3",
        "--agent", "researcher", "--resume", str(root), "--concurrency", "8"])
    config = {k: str(v) if isinstance(v, run.Path) else v for k, v in vars(args).items()}
    config["concurrency"] = 2
    manifest = {"run_id": "run", "status": "interrupted", "started_at": run.now(),
        "finished_at": run.now(), "assistant_id": "researcher", "tasks": ["a"],
        "repetitions": 3, "config": config, "provenance": {
            "revision": run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.REPO_ROOT, text=True).strip(),
            "data_sha256": run.hashlib.sha256(data.read_bytes()).hexdigest(), "backend": args.retrieval_url,
            "corpus_revision": run.CORPUS_REVISION,
            "model": run.AGENTS[args.agent][0], "worker_model": None, "worker_budget_seconds": None, "worker_grace_seconds": None,
        }, "episodes": [{"task": "a", "repetition": 1,
            "status": "finished", "evaluation": {"score": 1, "passed": True}}]}
    if mismatch == "agent":
        manifest["assistant_id"] = "decomposer"
    elif mismatch == "timeout":
        manifest["config"]["timeout"] = 100
    elif mismatch == "running":
        manifest["status"] = "running"
    run.save(root / "manifest.json", manifest)
    monkeypatch.setattr(run, "preflight", AsyncMock())
    calls = []
    async def spawn(*command, **kwargs):
        calls.append(command)
        directory = run.Path(command[command.index("--output-dir") + 1])
        assert not (directory / "model_calls.jsonl").exists()
        return SimpleNamespace(returncode=1, wait=AsyncMock(return_value=1))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    if mismatch:
        with pytest.raises(ValueError, match="Resume"):
            asyncio.run(run.main(args))
        assert not calls
        assert (partial / "model_calls.jsonl").read_text() == 'interrupted request'
        return
    assert asyncio.run(run.main(args)) == root
    result = json.loads((root / "manifest.json").read_text())
    assert len(calls) == 2
    assert len(result["episodes"]) == 3
    assert {(x["task"], x["repetition"]) for x in result["episodes"]} == {("a", 1), ("a", 2), ("a", 3)}
    assert result["episodes"][0] == manifest["episodes"][0]
    assert (complete / "trace.json").read_text() == 'original trace'
    assert next((root / "interrupted").glob('*/model_calls.jsonl')).read_text() == 'interrupted request'
    assert result["resumptions"][-1]["concurrency"] == 8
    assert result["status"] == "needs_attention"


@pytest.mark.parametrize('output,score,reason', [
    ('  New York\n', 1, 'exact_match'),
    ('\n  ', 0, 'missing_final_answer'),
])
def test_plain_answer_and_empty_response_need_no_judge(monkeypatch, tmp_path, output, score, reason):
    factory = Mock()
    monkeypatch.setattr(evaluate, 'create_model', factory)
    result = asyncio.run(evaluate.evaluate({'question': 'Which city?', 'answer': 'New York'}, output, tmp_path))
    assert result['score'] == score
    assert result['reason'] == reason
    factory.assert_not_called()


@pytest.mark.parametrize('output,verdict', [
    ('The answer is New York.', True),
    ('New York or Boston; I cannot determine which.', False),
    ('<tool_call><function=search>New York</function></tool_call>', False),
])
def test_entire_final_text_is_sent_to_judge(monkeypatch, tmp_path, output, verdict):
    model = SimpleNamespace(bind=lambda **kwargs: model,
                            ainvoke=AsyncMock(return_value=AIMessage(content=json.dumps({'match': verdict}))))
    monkeypatch.setattr(evaluate, 'create_model', lambda _: model)
    result = asyncio.run(evaluate.evaluate({'question': 'Which city?', 'answer': 'New York'}, output, tmp_path))
    assert result['passed'] is verdict
    messages = model.ainvoke.call_args.args[0]
    assert json.loads(messages[1]['content'])['generated'] == output
    assert json.loads((tmp_path / 'judge.json').read_text())['messages'] == messages
    assert result['format_ok'] is False
    assert result['answer'] == output


@pytest.mark.parametrize('output,answer,format_ok', [
    (r'Candidate Boston. \boxed{New York} More explanation.', 'New York', True),
    (r'\boxed{Boston} Correction: \boxed{New York}', 'New York', True),
    (r'Evidence. \boxed{fly\text{-}by} Caveat.', r'fly\text{-}by', True),
    (r'<think>\boxed{Boston}</think>New York', 'New York', False),
    (r'New York. \boxed{}', '', False),
    (r'Incomplete \boxed{New York', r'Incomplete \boxed{New York', False),
])
def test_boxed_answer_selection(monkeypatch, tmp_path, output, answer, format_ok):
    model = SimpleNamespace(bind=lambda **kwargs: model,
                            ainvoke=AsyncMock(return_value=AIMessage(content='{"match": false}')))
    monkeypatch.setattr(evaluate, 'create_model', lambda _: model)
    result = asyncio.run(evaluate.evaluate({'question': 'Which?', 'answer': 'other'}, output, tmp_path))
    assert result['answer'] == answer
    assert result['format_ok'] is format_ok
    if answer:
        assert json.loads(model.ainvoke.call_args.args[0][1]['content'])['generated'] == answer
    else:
        assert result['reason'] == 'missing_final_answer'
        model.ainvoke.assert_not_called()


def test_preflight_rejects_another_corpus_before_search(monkeypatch):
    calls = []
    def respond(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"corpus_revision": "another-index"})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(ValueError, match="Wiki-2018"):
        asyncio.run(run.preflight("http://retrieval"))
    assert calls == ["/health"]


def test_default_retrieval_is_wideseek():
    assert run.parser().parse_args([]).retrieval_url == "http://127.0.0.1:18080"
