import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langgraph_sdk import get_client, get_sync_client

from decomposer.core import _build_fork_tool, _build_run_tool, _build_wait_tool


@pytest.fixture
def subagent_server(tmp_path):
    config = tmp_path / "langgraph.json"
    graph_path = Path(__file__).parent / "fixtures" / "failing_subagent.py"
    conversation_path = Path(__file__).parent / "fixtures" / "conversation_subagent.py"
    config.write_text(json.dumps({
        "dependencies": ["."],
        "graphs": {
            "failing_subagent": f"{graph_path}:graph",
            **{name: f"{conversation_path}:{name}" for name in ("root", "worker")},
        },
    }))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable, "-m", "langgraph_cli", "dev",
                "--config", str(config), "--port", str(port),
                "--no-reload", "--no-browser",
                "--n-jobs-per-worker", "3",
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join([
                    str(Path(__file__).resolve().parents[1] / "src"),
                    str(Path(__file__).resolve().parents[1]),
                    os.environ.get("PYTHONPATH", ""),
                ]),
                "LANGGRAPH_CLI_NO_ANALYTICS": "1",

                "LANGSMITH_TRACING": "false",
                "LANGCHAIN_TRACING_V2": "false",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            with httpx.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        pytest.fail(log_path.read_text())
                    try:
                        if client.get(f"{url}/ok").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    pytest.fail(f"Subagent server did not start.\n{log_path.read_text()}")
            yield url
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def test_wait_receives_server_error(subagent_server) -> None:
    with get_sync_client(url=subagent_server, api_key=None, timeout=5) as client:
        thread_id = client.threads.create()["thread_id"]
        run = client.runs.create(
            thread_id,
            "failing_subagent",
            input={"messages": [{"role": "user", "content": "Call the failing tool."}]},
        )
        run_id = run["run_id"]
        deadline = time.monotonic() + 15
        while run["status"] in {"pending", "running"}:
            assert time.monotonic() < deadline, f"Run did not finish: {run}"
            time.sleep(0.1)
            run = client.runs.get(thread_id, run_id)

        runtime = SimpleNamespace(
            state={
                "subagents": {
                    thread_id: {
                        "subagent_id": thread_id,
                        "subagent_type_id": "failing_subagent",
                        "assistant_id": "failing_subagent",
                        "thread_id": thread_id,
                    },
                },
                "subagent_runs": {
                    run_id: {
                        "subagent_run_id": run_id,
                        "subagent_id": thread_id,
                        "run_id": run_id,
                        "status": "running",
                        "prompt": "Call the failing tool.",
                    },
                },
            },
            tool_call_id="wait_1",
        )
        tool = _build_wait_tool(SimpleNamespace(get_sync=lambda _: client))
        command = tool.func(runtime=runtime)
        completed_run = command.update["subagent_runs"][run_id]
    assert run["status"] == "error"
    assert completed_run["response"] is None
    assert json.loads(completed_run["error"]) == {
        "error": "RuntimeError", "message": "deliberate subagent tool failure",
    }
    assert json.loads(command.update["messages"][0].content) == [{
        "subagent_id": thread_id, "subagent_run_id": run_id,
        "status": "error", "response": None, "error": completed_run["error"],
    }]


def test_decomposer_reuses_worker_conversation(subagent_server) -> None:
    with get_sync_client(url=subagent_server, api_key=None, timeout=30) as client:
        root_id = client.threads.create()["thread_id"]
        previous_messages = {}
        previous_runs = {}
        previous_worker_id = None
        prompts = ["Remember 42.", "What did I ask you to remember?"]
        for turn, prompt in enumerate(prompts):
            run = client.runs.create(
                root_id, "root", input={"messages": [{"role": "user", "content": prompt}]},
            )
            client.runs.join(root_id, run["run_id"])
            assert client.runs.get(root_id, run["run_id"])["status"] == "success"
            root = client.threads.get_state(root_id)["values"]
            worker_id, = root["subagents"]
            assert root["subagents"][worker_id]["assistant_id"] == "worker"
            worker = client.threads.get_state(worker_id)["values"]
            if previous_worker_id is not None:
                assert worker_id == previous_worker_id
            previous_worker_id = worker_id

            expected = prompts[:turn + 1]
            for thread_id, state in ((root_id, root), (worker_id, worker)):
                messages = state["messages"]
                earlier = previous_messages.get(thread_id, [])
                assert messages[:len(earlier)] == earlier
                assert json.loads(messages[-1]["content"]) == expected
                previous_messages[thread_id] = messages

            runs = root["subagent_runs"]
            assert len(runs) == turn + 1
            assert {run_id: runs[run_id] for run_id in previous_runs} == previous_runs
            new_run_id, = runs.keys() - previous_runs.keys()
            new_run = runs[new_run_id]
            assert new_run["subagent_id"] == worker_id
            assert new_run["status"] == "responded"
            assert new_run["response"] == json.dumps(expected)
            assert new_run["error"] is None
            assert new_run["response_sequence_number"] == turn
            assert [m["type"] for m in new_run["messages"]] == ["human", "ai", "tool", "ai"]
            assert new_run["messages"][0]["content"] == prompts[turn]
            assert new_run["messages"][-1]["content"] == new_run["response"]
            previous_runs = runs


@pytest.mark.parametrize("async_invocation", [False, True])
def test_fork_preserves_history_and_branches_independently(subagent_server, async_invocation):
    with get_sync_client(url=subagent_server, api_key=None, timeout=30) as client:
        source_id = client.threads.create()["thread_id"]
        source = {
            "subagent_id": source_id, "thread_id": source_id,
            "subagent_type_id": "worker", "assistant_id": "worker",
        }
        runtime = SimpleNamespace(
            tool_call_id="run_source", context=None,
            state={"subagents": {source_id: source}, "subagent_runs": {}},
        )
        clients = SimpleNamespace(get_sync=lambda _: client)
        run_tool = _build_run_tool(clients, None)
        wait_tool = _build_wait_tool(clients)
        command = run_tool.func(source_id, "Remember 42.", runtime)
        runtime.state["subagent_runs"].update(command.update["subagent_runs"])
        source_run_id = json.loads(command.update["messages"][0].content)["subagent_run_id"]
        client.runs.join(source_id, source_run_id)
        command = wait_tool.func(runtime)
        runtime.state["subagent_runs"].update(command.update["subagent_runs"])
        source_run = runtime.state["subagent_runs"][source_run_id]
        assert source_run["status"] == "responded"
        assert source_run["tool_calls"] == [{
            "id": "echo_1", "name": "echo", "args": {"text": "Remember 42."},
        }]
        history = client.threads.get_state(source_id)["values"]["messages"]
        runtime.tool_call_id = "fork"
        async def fork():
            async with get_client(url=subagent_server, api_key=None, timeout=30) as async_client:
                tool = _build_fork_tool(SimpleNamespace(get_async=lambda _: async_client))
                return await tool.coroutine(source_id, runtime)

        if async_invocation:
            command = asyncio.run(fork())
        else:
            tool = _build_fork_tool(SimpleNamespace(get_sync=lambda _: client))
            command = tool.func(source_id, runtime)
        fork_id = json.loads(command.update["messages"][0].content)["subagent_id"]
        assert fork_id != source_id
        assert command.update["subagents"][fork_id] == {
            **source, "subagent_id": fork_id, "thread_id": fork_id,
        }
        assert "subagent_runs" not in command.update
        assert client.threads.get_state(fork_id)["values"]["messages"] == history
        runtime.state["subagents"].update(command.update["subagents"])
        runs = []
        for thread_id, prompt in ((source_id, "Original branch."), (fork_id, "Forked branch.")):
            runtime.tool_call_id = f"run_{thread_id}"
            command = run_tool.func(thread_id, prompt, runtime)
            runtime.state["subagent_runs"].update(command.update["subagent_runs"])
            run_id = json.loads(command.update["messages"][0].content)["subagent_run_id"]
            runs.append((thread_id, prompt, run_id))
        for thread_id, _, run_id in runs:
            client.runs.join(thread_id, run_id)

        runtime.tool_call_id = "wait_branches"
        command = wait_tool.func(runtime)
        assert set(command.update["subagent_runs"]) == {run_id for _, _, run_id in runs}
        summaries = json.loads(command.update["messages"][0].content)
        assert {summary["subagent_run_id"] for summary in summaries} == {
            run_id for _, _, run_id in runs
        }
        for thread_id, prompt, run_id in runs:
            run = command.update["subagent_runs"][run_id]
            assert run["status"] == "responded"
            assert json.loads(run["response"]) == ["Remember 42.", prompt]
            assert len(run["messages"]) == 4
            assert run["messages"][0]["content"] == prompt
            assert run["messages"][-1]["content"] == run["response"]
            assert run["tool_calls"] == [{
                "id": "echo_2", "name": "echo", "args": {"text": prompt},
            }]
            messages = client.threads.get_state(thread_id)["values"]["messages"]
            assert messages[:len(history)] == history
            assert json.loads(messages[-1]["content"]) == ["Remember 42.", prompt]
