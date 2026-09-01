import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client


MAX_TOOL_OUTPUT_CHARS = 8000
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
