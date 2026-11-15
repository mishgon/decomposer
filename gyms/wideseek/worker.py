"""The same offline search tools serve simple agents and delegated workers."""
import asyncio
import os
import time
from typing import Annotated
from uuid import uuid4

import httpx
from langchain.tools import tool, ToolRuntime
from pydantic import Field

from gyms.wideseek.runtime import Context, directory, save


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
    """Search Wiki-2018. Returns up to 1500 characters per passage and URLs; topk is 1–10.

    Use access(url, offset=...) to read beyond a truncated passage preview.
    """
    if not 1 <= topk <= 10:
        raise ValueError("topk must be between 1 and 10")
    docs = await request_tool("/retrieve", {"queries": [query], "topk": topk}, runtime)
    return "\n\n".join(
        f"[Doc {i+1}]({d['url']}):\n{d['contents'][:1500]}"
        + ("\n[Passage preview truncated; use access(url) for the full page.]"
           if len(d['contents']) > 1500 else "")
        for i, d in enumerate(docs)) or "No search results are found."


@tool
async def access(url: str, runtime: ToolRuntime[Context],
                 access_token: Annotated[int, Field(ge=1, le=6000)] = 3000,
                 offset: Annotated[int, Field(ge=0)] = 0) -> str:
    """Read a Wiki-2018 page slice. access_token is 1–6000 characters, not tokens.

    offset is a character offset into the full page. Read another slice only
    when a required fact is missing. Stop once enough evidence is available.
    """
    if not 1 <= access_token <= 6000 or offset < 0:
        raise ValueError("access_token must be between 1 and 6000; offset must be nonnegative")
    page = await request_tool("/access", {"urls": [url]}, runtime)
    if not page:
        return "[Page not found. Use an exact URL returned by search; do not retry this unchanged URL.]"
    end = min(offset + access_token, len(page["contents"]))
    if offset >= len(page["contents"]):
        return "[End of page.]"
    text = page["contents"][offset:end]
    return (text + f"\n[Page truncated; next offset={end}. Use access with this offset to continue.]"
            if end < len(page["contents"]) else text + "\n[End of page.]")
