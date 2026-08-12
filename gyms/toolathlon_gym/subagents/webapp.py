import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_mcp_adapters.tools import load_mcp_tools


MAX_TOOL_OUTPUT_CHARS = 8000


@wrap_tool_call
async def truncate_mcp_tool_output(request, handler):
    response = await handler(request)
    metadata = request.tool.metadata if request.tool else None
    if isinstance(response, ToolMessage) and metadata and metadata.get("toolathlon_mcp"):
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            response = response.model_copy(
                update={
                    "content": content[:MAX_TOOL_OUTPUT_CHARS]
                    + f"\n...[truncated, total {len(content)} chars]"
                }
            )
    return response


def _resolve(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in replacements.items():
            value = value.replace(f"${{{key}}}", replacement)
    elif isinstance(value, list):
        value = [_resolve(item, replacements) for item in value]
    elif isinstance(value, dict):
        value = {
            key: _resolve(item, replacements) for key, item in value.items()
        }
    return value


def load_connections(
    task_config: dict[str, Any],
    toolathlon_root: Path,
) -> dict[str, StdioConnection]:
    needed_servers = task_config["needed_mcp_servers"]
    workspace = str(Path(task_config["agent_workspace"]).resolve())
    task_dir = str(
        (toolathlon_root / "tasks" / "finalpool" / task_config["task_dir"]).resolve()
    )
    local_servers = os.environ.get(
        "LOCAL_SERVERS_PATH",
        str((toolathlon_root / "local_servers").resolve()),
    )
    replacements = {
        "local_servers_paths": local_servers,
        "agent_workspace": workspace,
        "task_dir": task_dir,
    }

    connections: dict[str, StdioConnection] = {}
    config_dir = toolathlon_root / "configs" / "mcp_servers"
    for path in sorted(config_dir.glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not config:
            continue
        name = config.get("name", path.stem)
        if name not in needed_servers:
            continue
        if config.get("type", "stdio") != "stdio":
            raise ValueError(f"Unsupported MCP transport for {name!r}")

        params = _resolve(config.get("params", {}), replacements)
        env = {**params.get("env", {}), **os.environ}
        pg_env = {
            "PGHOST": "PG_HOST",
            "PGPORT": "PG_PORT",
            "PGDATABASE": "PG_DATABASE",
            "PGUSER": "PG_USER",
            "PGPASSWORD": "PG_PASSWORD",
        }
        env.update(
            {target: env[source] for source, target in pg_env.items() if source in env}
        )
        params["env"] = env
        params.setdefault("cwd", workspace)
        Path(params["cwd"]).mkdir(parents=True, exist_ok=True)

        timeout = float(config.get("client_session_timeout_seconds", 60))
        connections[name] = {
            "transport": "stdio",
            **params,
            "session_kwargs": {
                "read_timeout_seconds": timedelta(seconds=timeout),
            },
        }

    missing = set(needed_servers) - connections.keys()
    if missing:
        raise ValueError(f"Missing MCP configurations: {sorted(missing)}")
    return connections


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    data_dir = Path(os.environ.get("TOOLATHLON_DATA_DIR", "/artifacts/data"))
    toolathlon_root = Path(os.environ.get("TOOLATHLON_ROOT", "/workspace"))
    task_config = json.loads(
        (data_dir / "runtime.json").read_text(encoding="utf-8")
    )["task_config"]
    connections = load_connections(task_config, toolathlon_root)
    client = MultiServerMCPClient(connections)

    async with AsyncExitStack() as stack:
        tools: list[BaseTool] = []
        needed_local_tools = task_config["needed_local_tools"]
        if {"python_execute", "handle_overlong_tool_outputs"} & set(
            needed_local_tools
        ):
            sys.path.insert(0, str(toolathlon_root))
            from utils.aux_tools.overlong_tool_manager import make_overlong_tools
            from utils.aux_tools.python_interpretor import make_python_execute

        if "python_execute" in needed_local_tools:
            native_python_execute = make_python_execute(
                task_config["agent_workspace"]
            )

            def python_execute(
                code: str,
                filename: str = "",
                timeout: int = 30,
            ) -> str:
                """Execute Python code in the Toolathlon task workspace."""
                coroutine = native_python_execute(code, filename, timeout)
                try:
                    coroutine.send(None)
                except StopIteration as result:
                    return result.value
                coroutine.close()
                raise RuntimeError("Toolathlon python_execute unexpectedly awaited")

            tools.append(tool(python_execute))
        if "handle_overlong_tool_outputs" in needed_local_tools:
            tools.extend(
                tool(fn)
                for fn in make_overlong_tools(task_config["agent_workspace"])
            )
        for server_name in connections:
            session = await stack.enter_async_context(client.session(server_name))
            mcp_tools = await load_mcp_tools(session, server_name=server_name)
            for mcp_tool in mcp_tools:
                mcp_tool.metadata = {
                    **(mcp_tool.metadata or {}),
                    "toolathlon_mcp": True,
                }
            tools.extend(mcp_tools)
        app.state.tools = tools
        try:
            yield
        finally:
            del app.state.tools


app = FastAPI(lifespan=lifespan)


def get_tools() -> list[BaseTool]:
    tools = getattr(app.state, "tools", None)
    if tools is None:
        raise RuntimeError("Subagent tools have not started")
    return tools.copy()
