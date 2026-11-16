"""Agentic-rag search/fetch with checkpointed, per-agent URL access."""
import asyncio
import json
from pathlib import Path
import os
from typing import Annotated

import httpx
from gyms.model_logging import append_record
from langchain.agents import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command


def merge_pages(left, right):
    return {**left, **right}


class ResearchState(AgentState):
    pages: Annotated[dict[str, str], merge_pages]


async def request(endpoint, payload):
    try:
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            response = await client.post(
                os.environ["BROWSECOMP_RETRIEVAL_URL"].rstrip("/") + endpoint,
                json={**payload, "source": "browsecomp_plus"},
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as error:
        path = Path(os.environ["BROWSECOMP_MODEL_LOG"]).with_name("web_calls.jsonl")
        await asyncio.to_thread(append_record, path, {"endpoint": endpoint, "error": repr(error)})
        raise


@tool
async def search(query: str, runtime: ToolRuntime, limit: int = 5) -> Command:
    """Search the BrowseComp-Plus corpus. Returns URLs, titles and snippets."""
    hits = (await request("/search", {"query": query, "limit": limit}))["results"]
    pages = {hit["url"]: hit["page_id"] for hit in hits}
    content = {"results": [{key: hit[key] for key in ("url", "title", "snippet")} for hit in hits]}
    return Command(update={"pages": pages, "messages": [
        ToolMessage(content=json.dumps(content), tool_call_id=runtime.tool_call_id)
    ]})


@tool
async def fetch(url: str, runtime: ToolRuntime) -> dict:
    """Fetch the body of a URL returned by this agent's searches."""
    pages = runtime.state.get("pages", {})
    if url not in pages:
        return {"error": "url_not_in_session", "message": "Search for this URL first."}
    return await request("/fetch", {"page_id": pages[url]})
