import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client


# Match Toolathlon-Verified's own scaffold.  The previous 8K limit discarded
# most of many search/list results long before the benchmark's 100K cutoff.
MAX_TOOL_OUTPUT_CHARS = int(
    os.environ.get("TOOLATHLON_MAX_TOOL_OUTPUT_CHARS", "100000")
)
GATEWAY_SSE_READ_TIMEOUT_SECONDS = 30 * 60
GATEWAY_TOOL_READ_TIMEOUT_SECONDS = 300


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
            output_id = uuid.uuid4().hex
            relative_path = f".overlong_tool_outputs/{output_id}.json"
            workspace = os.environ.get("TOOLATHLON_AGENT_WORKSPACE")
            saved_note = ""
            if workspace:
                output_path = Path(workspace) / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(content, encoding="utf-8")
                saved_note = (
                    f" The complete output is available at {relative_path}."
                )
            response = response.model_copy(
                update={
                    "content": content[:MAX_TOOL_OUTPUT_CHARS]
                    + f"\n...[truncated, total {len(content)} chars]."
                    + saved_note
                }
            )
    return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    gateway_url = os.environ.get("TOOLATHLON_GATEWAY_URL")
    if not gateway_url:
        raise RuntimeError("Set TOOLATHLON_GATEWAY_URL to the container gateway SSE endpoint.")

    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(
            sse_client(
                gateway_url,
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
        tools: list[BaseTool] = await load_mcp_tools(
            session, server_name="gateway"
        )
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
