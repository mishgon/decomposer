import asyncio
import json
import sys
import threading
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage


SUBAGENTS_DIR = Path(__file__).parents[1] / "gyms" / "toolathlon_gym" / "subagents"
TOOLATHLON_ROOT = Path(__file__).parents[1] / "external" / "toolathlon_gym"
sys.path.insert(0, str(SUBAGENTS_DIR))

import webapp


@pytest.mark.parametrize("script", [False, True])
def test_uv_mcp_launch_uses_preinstalled_environment(tmp_path, monkeypatch, script):
    import yaml
    root = tmp_path / "gym"
    config_dir = root / "configs/mcp_servers"
    config_dir.mkdir(parents=True)
    project = tmp_path / "servers/example"
    binary = project / ".venv/bin" / ("python" if script else "example-mcp")
    binary.parent.mkdir(parents=True)
    binary.touch()
    arguments = (["run", str(project / "server.py")] if script else
                 ["--directory", str(project), "run", "example-mcp", "--stdio"])
    (config_dir / "example.yaml").write_text(yaml.safe_dump({
        "name": "example", "params": {"command": "uv", "args": arguments},
    }))
    config = {"needed_mcp_servers": ["example"], "agent_workspace": str(tmp_path / "work"),
              "task_dir": "fixture"}
    connection = webapp.load_connections(config, root)["example"]
    assert connection["command"] == str(binary)
    assert connection["args"] == ([str(project / "server.py")] if script else ["--stdio"])
    assert connection["env"]["VIRTUAL_ENV"] == str(project / ".venv")
    binary.unlink()
    with pytest.raises(RuntimeError, match="Preinstalled MCP executable missing"):
        webapp.load_connections(config, root)


def test_load_connections(tmp_path: Path, monkeypatch) -> None:
    toolathlon_root = tmp_path / "toolathlon"
    config_dir = toolathlon_root / "configs" / "mcp_servers"
    config_dir.mkdir(parents=True)
    (config_dir / "test.yaml").write_text(
        """
name: test
type: stdio
params:
  command: python
  args:
    - ${local_servers_paths}/server.py
    - ${task_dir}
  env:
    PG_HOST: configured
    WORKSPACE: ${agent_workspace}
  cwd: ${agent_workspace}
client_session_timeout_seconds: 42
""",
        encoding="utf-8",
    )

    workspace = tmp_path / "workspace"
    monkeypatch.setenv("PGHOST", "database")

    connection = webapp.load_connections(
        {
            "task_dir": "example",
            "agent_workspace": str(workspace),
            "needed_mcp_servers": ["test"],
        },
        toolathlon_root,
    )["test"]

    assert connection["command"] == "python"
    assert connection["args"] == [
        str(toolathlon_root / "local_servers" / "server.py"),
        str(toolathlon_root / "tasks" / "finalpool" / "example"),
    ]
    assert connection["cwd"] == str(workspace)
    assert connection["env"]["PG_HOST"] == "database"
    assert connection["env"]["WORKSPACE"] == str(workspace)
    assert connection["session_kwargs"] == {
        "read_timeout_seconds": timedelta(seconds=42)
    }


def test_webapp_keeps_sessions_open(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    connections = {
        "first": {"transport": "stdio", "command": "first", "args": []},
        "second": {"transport": "stdio", "command": "second", "args": []},
    }

    class FakeClient:
        def __init__(self, configured_connections) -> None:
            assert configured_connections == connections

        @asynccontextmanager
        async def session(self, server_name):
            events.append(f"open:{server_name}")
            try:
                yield server_name
            finally:
                events.append(f"close:{server_name}")

    async def fake_load_tools(session, *, server_name):
        assert session == server_name
        events.append(f"load:{server_name}")
        return [SimpleNamespace(name=f"{server_name}_tool", metadata=None)]

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "runtime.json").write_text(
        json.dumps(
            {
                "task_config": {
                    "agent_workspace": str(tmp_path / "workspace"),
                    "needed_local_tools": [],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TOOLATHLON_DATA_DIR", str(data_dir))

    monkeypatch.setattr(
        webapp,
        "load_connections",
        lambda task_config, toolathlon_root: connections,
    )
    monkeypatch.setattr(webapp, "MultiServerMCPClient", FakeClient)
    monkeypatch.setattr(webapp, "load_mcp_tools", fake_load_tools)

    async def run() -> None:
        async with webapp.lifespan(webapp.app):
            tools = webapp.get_tools()
            assert [tool.name for tool in tools] == ["first_tool", "second_tool"]
            assert all(tool.metadata == {"toolathlon_mcp": True} for tool in tools)
            assert events == [
                "open:first",
                "load:first",
                "open:second",
                "load:second",
            ]

        assert events[-2:] == ["close:second", "close:first"]
        with pytest.raises(RuntimeError, match="have not started"):
            webapp.get_tools()

    asyncio.run(run())


def test_mcp_tool_outputs_are_truncated() -> None:
    async def run() -> None:
        output = "x" * (webapp.MAX_TOOL_OUTPUT_CHARS + 1)

        async def handler(request):
            return ToolMessage(content=output, tool_call_id="call")

        request = SimpleNamespace(
            tool=SimpleNamespace(metadata={"toolathlon_mcp": True})
        )
        response = await webapp.truncate_mcp_tool_output.awrap_tool_call(
            request,
            handler,
        )

        assert response.content == (
            output[:webapp.MAX_TOOL_OUTPUT_CHARS]
            + f"\n...[truncated, total {len(output)} chars]"
        )

        request.tool.metadata = None
        response = await webapp.truncate_mcp_tool_output.awrap_tool_call(
            request,
            handler,
        )
        assert response.content == output

    asyncio.run(run())


def test_webapp_loads_native_local_tools(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "runtime.json").write_text(
        json.dumps(
            {
                "task_config": {
                    "agent_workspace": str(tmp_path / "workspace"),
                    "needed_local_tools": [
                        "python_execute",
                        "handle_overlong_tool_outputs",
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TOOLATHLON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("TOOLATHLON_ROOT", str(TOOLATHLON_ROOT))
    monkeypatch.setattr(
        webapp,
        "load_connections",
        lambda task_config, toolathlon_root: {},
    )
    sys.path.insert(0, str(TOOLATHLON_ROOT))
    from gyms.toolathlon_gym.subagents import python_execute as python_module

    def make_python_execute(workspace):
        def python_execute(
            code: str,
            filename: str = "",
            timeout: int = 30,
        ) -> str:
            """Return executor thread identity for this test."""
            return str(threading.get_ident())

        return python_execute

    monkeypatch.setattr(
        python_module,
        "make_python_execute",
        make_python_execute,
    )
    event_loop_thread = str(threading.get_ident())

    async def run() -> None:
        async with webapp.lifespan(webapp.app):
            tools = webapp.get_tools()
            assert [tool.name for tool in tools] == [
                "python_execute",
                "save_overlong_output",
                "view_overlong_output",
            ]
            assert await tools[0].ainvoke({"code": "pass"}) != event_loop_thread

    asyncio.run(run())
