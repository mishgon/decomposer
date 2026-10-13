import asyncio
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

import pytest
from langgraph_sdk import get_client, get_sync_client

from decomposer.core import _build_run_tool
from decomposer.prompts import AGENT_GRACEFUL_SHUTDOWN_REQUEST


@pytest.fixture(scope="module")
def agent_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("budget-server")
    module = directory / "agents.py"
    module.write_text('''
import asyncio
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from decomposer.run_budget import RunBudgetContext, RunBudgetMiddleware

class DictContext(RunBudgetContext):
    value: str

@dataclass
class DataclassContext:
    value: str
    agent_run_budget_seconds: float | None = None
    agent_shutdown_grace_seconds: float = 60

class StrictContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str
    agent_run_budget_seconds: float | None = None
    agent_shutdown_grace_seconds: float = 60

class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

def model():
    return Model(responses=[
        AIMessage(content="", tool_calls=[{"name":"work", "args":{}, "id":"work"}]),
        AIMessage(content="Partial result"),
    ])

@tool(description="Perform part of the task.")
async def work() -> str:
    await asyncio.sleep(.1)
    return "work done"

def make(schema):
    return create_agent(model(), tools=[work], context_schema=schema,
                        middleware=[RunBudgetMiddleware()])

def dictionary(): return make(DictContext)
def dataclass_agent(): return make(DataclassContext)
def pydantic_agent(): return make(StrictContext)

def gaia():
    from gyms.gaia2 import agents
    agents._model = model
    agents._worker_tools = lambda context, consumer: [work]
    return agents.qwen_3_5_4b_thinking

if __name__ == "__main__":
    import sys
    from langgraph_api.cli import run_server
    run_server(host="127.0.0.1", port=int(sys.argv[1]),
        graphs={name: f"./agents.py:{factory}" for name, factory in {
            "dictionary":"dictionary", "dataclass":"dataclass_agent",
            "pydantic":"pydantic_agent", "gaia":"gaia",
        }.items()}, open_browser=False, reload=False,
        disable_persistence=True, allow_blocking=True)
''')
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = Path(__file__).resolve().parents[1]
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join(map(str, (root / "src", root, root / "external/gaia2"))),
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
    }
    url = f"http://127.0.0.1:{port}"
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(module), str(port)], cwd=directory,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail((directory / "server.log").read_text())
                try:
                    with urllib.request.urlopen(url + "/ok", timeout=1):
                        break
                except OSError:
                    time.sleep(.1)
            else:
                pytest.fail("Agent Server did not start")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.parametrize("assistant", ["dictionary", "dataclass", "pydantic", "gaia"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("budget", [None, .05])
def test_run_budget_through_agent_server(agent_server, assistant, asynchronous, budget):
    client = get_sync_client(url=agent_server)
    thread = client.threads.create()
    context = {"worker_system_prompt": "Work carefully."} if assistant == "gaia" else {"value": "environment"}
    runtime = SimpleNamespace(context=context, tool_call_id="run", state={
        "agents": {"agent": {"agent_type_id": "test", "thread_id": thread["thread_id"],
                               "assistant_id": assistant, "status": "idle"}},
        "agent_runs": {},
    })
    clients = SimpleNamespace(get_sync=lambda _: client, get_async=lambda _: get_client(url=agent_server))
    run = _build_run_tool(clients, 100, budget, 5)
    try:
        if asynchronous:
            asyncio.run(run.coroutine("agent", "Do work", runtime))
        else:
            run.func("agent", "Do work", runtime)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            runs = client.runs.list(thread["thread_id"])
            if runs and runs[0]["status"] not in {"pending", "running"}:
                break
            time.sleep(.1)
        assert runs[0]["status"] == "success", client.threads.get_state(thread["thread_id"])
        messages = client.threads.get_state(thread["thread_id"])["values"]["messages"]
        assert messages[-1]["content"] == "Partial result"
        assert (messages[-2]["content"] == AGENT_GRACEFUL_SHUTDOWN_REQUEST) == (budget is not None)
        assert any(m["type"] == "tool" and m["content"] == "work done" for m in messages)
        assert "agent_run_budget_seconds" not in context
    finally:
        client.threads.delete(thread["thread_id"])
