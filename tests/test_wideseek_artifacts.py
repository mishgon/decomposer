import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from gyms.model_logging import append_record
from gyms.wideseek.run import run_jobs, usage
from gyms.wideseek.runtime import ModelLog, init_budget


def test_incremental_calls_preserve_reasoning_and_count_once(tmp_path):
    path = tmp_path / "episode"
    init_budget(path)
    response = AIMessage(content="answer", additional_kwargs={"reasoning_content": "reasoning"},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                        "output_token_details": {"reasoning": 3}})
    request = SimpleNamespace(runtime=SimpleNamespace(context={"directory": str(path)}),
        model=SimpleNamespace(), model_settings={}, tools=[], system_message=None,
        messages=[HumanMessage(content="question")])
    request.override = lambda **kwargs: request
    handler = AsyncMock(return_value=SimpleNamespace(result=[response]))
    middleware = ModelLog("react")
    with patch("gyms.wideseek.runtime.get_config", return_value={"configurable": {"thread_id": "thread"}}):
        asyncio.run(middleware.awrap_model_call(request, handler))
        request.messages += [response, ToolMessage(content="tool result", tool_call_id="tool")]
        asyncio.run(middleware.awrap_model_call(request, handler))
        handler.side_effect = TimeoutError("provider timeout")
        try:
            asyncio.run(middleware.awrap_model_call(request, handler))
        except TimeoutError:
            pass
    records = [json.loads(line) for line in (path / "model_calls.jsonl").read_text().splitlines()]
    assert len(records) == 6
    assert records[0]["thread_id"] == "thread"
    assert records[2]["request_message_count"] == 3
    assert [m["data"]["content"] for m in records[2]["request_delta"]] == ["tool result"]
    assert records[1]["response"][0]["data"]["additional_kwargs"]["reasoning_content"] == "reasoning"
    append_record(path / "model_calls.jsonl", {"call_id": "unfinished", "role": "react", "status": "started"})
    totals = usage(path)["react"]
    assert (totals["calls"], totals["errors"], totals["unfinished"]) == (4, 1, 1)
    assert (totals["input_tokens"], totals["output_tokens"], totals["reasoning_tokens"]) == (20, 10, 6)


def test_concurrent_jsonl_appends(tmp_path):
    path = tmp_path / "calls.jsonl"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda n: append_record(path, {"call_id": n, "text": "x" * 4096}), range(100)))
    assert {json.loads(line)["call_id"] for line in path.read_text().splitlines()} == set(range(100))


def test_episode_manifest_and_html(tmp_path):
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={
        "messages": [HumanMessage(content="question"), AIMessage(content="answer")],
        "decomposer_agent_runs": [{"started_at": 1., "collected_at": 2., "status": "success",
                                   "agent_id": "controller", "agent_run_id": "controller-run",
                                   "prompt": "question", "response": "answer"}]})
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={}))
    args = SimpleNamespace(agent="decomposer", mode="decomposer", concurrency=2,
                           model_calls=None, output_tokens=None, worker_url="http://unused", episode_timeout=1)
    task = {"task_id": "task", "question": "question", "answer": "answer", "unique_columns": []}
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": []}))
    with patch("gyms.wideseek.run.model", return_value=SimpleNamespace()), \
            patch("gyms.wideseek.run.agents.decomposer", return_value=graph), \
            patch("gyms.wideseek.run.evaluate", new=AsyncMock(return_value={"status": "scored", "score": .5})):
        asyncio.run(run_jobs([task], [("task", 1), ("task", 2)], tmp_path, args))
        asyncio.run(run_jobs([task], [("task", 1)], tmp_path, args))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert len(manifest["episodes"]) == 2
    for episode in manifest["episodes"]:
        path = tmp_path / episode["trace_path"]
        trace = json.loads(path.read_text())
        assert trace["episode_id"] == episode["episode_id"]
        assert trace["task"] == "task"
        assert trace["messages"][-1]["data"]["content"] == "answer"
        assert path.with_suffix(".html").exists()
        assert (path.parent / "usage.json").exists()


def test_logging_inside_agent_graph(tmp_path):
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    from gyms.wideseek.runtime import Context

    path = tmp_path / "episode"
    init_budget(path)
    graph = create_agent(FakeListChatModel(responses=["answer"]),
                         context_schema=Context, middleware=[ModelLog("react")],
                         checkpointer=InMemorySaver())
    state = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="question")]},
        config={"configurable": {"thread_id": "real-graph"}}, context={"directory": str(path)}))
    assert state["messages"][-1].content == "answer"
    records = [json.loads(line) for line in (path / "model_calls.jsonl").read_text().splitlines()]
    assert records[0]["thread_id"] == "real-graph"
    assert records[1]["status"] == "success"


def test_episode_setup_failure_does_not_interrupt_batch(tmp_path):
    from gyms.wideseek import run
    from gyms.wideseek.runtime import save

    save(tmp_path / 'manifest.json', {'episodes': []})
    tasks = [{'task_id': name} for name in ('broken', 'healthy')]
    args = SimpleNamespace(mode='decomposer', concurrency=1)

    async def execute(task, mode, attempt, root, args, **kwargs):
        if task['task_id'] == 'broken':
            raise ValueError('setup failed')
        save(root / mode / task['task_id'] / 'attempt-001/result.json', {
            'episode_id': 'healthy', 'task_id': 'healthy', 'task': 'healthy',
            'attempt': 1, 'repetition': 1, 'status': 'finished', 'started_at': 0,
            'finished_at': 1, 'trace_path': 'trace.json', 'evaluation': {'score': 1.},
        })

    with patch.object(run, 'episode', side_effect=execute):
        asyncio.run(run_jobs(tasks, [(task['task_id'], 1) for task in tasks], tmp_path, args))
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert [row['status'] for row in manifest['episodes']] == ['error', 'finished']
    failed = json.loads((tmp_path / 'decomposer/broken/attempt-001/result.json').read_text())
    assert failed['error'] == 'ValueError: setup failed'
    assert failed['evaluation']['score'] is None
