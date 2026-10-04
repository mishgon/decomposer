import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client


# Toolathlon's own scaffold truncates tool outputs at 100K characters.
MAX_TOOL_OUTPUT_CHARS = 100_000
GATEWAY_SSE_READ_TIMEOUT_SECONDS = 30 * 60
GATEWAY_TOOL_READ_TIMEOUT_SECONDS = 300


def _write_overlong_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@wrap_tool_call
async def truncate_mcp_tool_output(request, handler):
    try:
        response = await handler(request)
    except Exception as error:
        return ToolMessage(
            content=f"Tool call failed: {error}",
            tool_call_id=request.tool_call["id"],
            status="error",
        )
    if isinstance(response, ToolMessage):
        content = (
            response.content
            if isinstance(response.content, str)
            else json.dumps(response.content, ensure_ascii=False, default=str)
        )
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            # Toolathlon's overlong-output tools read the full text back from here.
            output_id = uuid.uuid4().hex
            relative_path = f".overlong_tool_outputs/{output_id}.json"
            workspace = Path(get_runtime()["agent_workspace"])
            await asyncio.to_thread(
                _write_overlong_output, workspace / relative_path, content
            )
            response = response.model_copy(
                update={
                    "content": content[:MAX_TOOL_OUTPUT_CHARS]
                    + f"\n...[truncated, total {len(content)} chars]."
                    + " The complete output is available through the overlong-output "
                    f"tools with shortuuid identifier {output_id}, and at {relative_path}."
                }
            )
    return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    data_dir = Path(os.environ.get("TOOLATHLON_DATA_DIR", "/workspace/dumps"))
    runtime = json.loads((data_dir / "runtime.json").read_text(encoding="utf-8"))

    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(
            sse_client(
                runtime["gateway_url"],
                sse_read_timeout=GATEWAY_SSE_READ_TIMEOUT_SECONDS,
            )
        )
        session = await stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(
                    seconds=GATEWAY_TOOL_READ_TIMEOUT_SECONDS
                ),
            )
        )
        await session.initialize()
        app.state.runtime = runtime
        app.state.tools = await load_mcp_tools(session, server_name="gateway")
        try:
            yield
        finally:
            del app.state.tools
            del app.state.runtime


app = FastAPI(lifespan=lifespan)


def get_tools() -> list[BaseTool]:
    tools = getattr(app.state, "tools", None)
    if tools is None:
        raise RuntimeError("Agent tools have not started")
    return tools.copy()


def get_runtime() -> dict[str, Any]:
    """Return the episode runtime read before any agent run could modify it."""
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Agent tools have not started")
    return runtime
