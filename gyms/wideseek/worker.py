"""The same offline search tools serve simple agents and delegated workers."""
import asyncio
import os
import time
from typing import Annotated
from uuid import uuid4

import httpx
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from pydantic import Field

from gyms.wideseek.runtime import Context, ModelLog, directory, model, save


async def request_tool(endpoint, payload, runtime):
    row = {"tool": endpoint, "arguments": payload, "started_at": time.time()}
    path = (await asyncio.to_thread(directory, runtime.context)) / "tool_calls" / f"{uuid4().hex}.json"
    try:
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            response = await client.post(os.environ.get("WS_SEARCH_URL", "http://127.0.0.1:18080") + endpoint,
                                         json=payload)
            response.raise_for_status()
            row["response"] = response.json()
            return row["response"]["result"][0]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        row["finished_at"] = time.time()
        await asyncio.to_thread(save, path, row)


@tool
async def search(query: str, runtime: ToolRuntime[Context],
                 topk: Annotated[int, Field(ge=1, le=10)] = 3) -> str:
    """Search the fixed Wiki-2018 corpus. Returns passages and URLs; topk is 1–10."""
    if not 1 <= topk <= 10:
        raise ValueError("topk must be between 1 and 10")
    docs = await request_tool("/retrieve", {"queries": [query], "topk": topk}, runtime)
    return "\n\n".join(f"[Doc {i+1}]({d['url']}):\n{d['contents'][:5000]}" for i, d in enumerate(docs)) or "No search results are found."


@tool
async def access(url: str, runtime: ToolRuntime[Context],
                 access_token: Annotated[int, Field(ge=1, le=20000)] = 5000) -> str:
    """Read a Wiki-2018 page by URL. access_token limits characters, from 1 to 20000."""
    if not 1 <= access_token <= 20000:
        raise ValueError("access_token must be between 1 and 20000")
    page = await request_tool("/access", {"urls": [url]}, runtime)
    return page["contents"][:access_token] if page else "No More Information is Found for this URL."


SYSTEM_PROMPT = (
    "Answer the user's research task using the provided offline search and access tools. "
    "The corpus is Wikipedia from 2018. Follow the requested output format. "
    "Do not invent facts or claim to have checked sources you did not read.")


def graph():
    return create_agent(model(), tools=[search, access], context_schema=Context,
        middleware=[ModelLog("researcher")], system_prompt=SYSTEM_PROMPT)
