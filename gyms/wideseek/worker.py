"""The same offline search tools serve simple agents and delegated workers."""
import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

import httpx
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import message_to_dict, convert_to_messages
from pydantic import BaseModel, Field

from external.wideseek_tools.prompt import get_access_summary_messages
from gyms.model_logging import append_record
from gyms.wideseek.runtime import Context, directory, save, model, model_metadata, close_model, MODEL_BACKEND


SUMMARY_MODEL = ("vllm/qwen_3_5_4b_non_thinking" if MODEL_BACKEND == "local"
                 else "lmrouter/qwen_3_5_4b_non_thinking")
ACCESS_CHARACTERS = 6000


async def summarize_access(info_to_extract, content, runtime):
    # Like upstream's judge-backed extraction, this is an auxiliary tool call,
    # outside the policy-call budget. Record its full cost separately by role.
    messages = get_access_summary_messages(info_to_extract, content)
    policy = model(SUMMARY_MODEL)
    log = directory(runtime.context) / "model_calls.jsonl"
    call_id = uuid4().hex
    started = time.monotonic()
    settings = {"temperature": 0., "presence_penalty": 0., "max_tokens": 2048}
    row = {"call_id": call_id, "role": "access_summary", "model": SUMMARY_MODEL,
           "thread_id": runtime.config.get("configurable", {}).get("thread_id"),
           "started_at": datetime.now(timezone.utc).isoformat(), "status": "started",
           "budget_scope": "auxiliary", "max_tokens": settings["max_tokens"],
           "generation": {**model_metadata(policy), **settings},
           "request_message_count": len(messages),
           "request_delta": [message_to_dict(m) for m in convert_to_messages(messages)]}
    outcome = {"status": "error", "error": "Summary did not complete"}
    try:
        await asyncio.to_thread(append_record, log, row)
        response = await policy.ainvoke(messages, **settings)
        if not response.text.strip():
            raise ValueError("Access summarizer returned no text")
        outcome = {"status": "success", "response": [message_to_dict(response)]}
        return response.text
    except BaseException as exc:
        outcome = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        try:
            await asyncio.to_thread(append_record, log, {
                "call_id": call_id, "role": "access_summary", **outcome,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - started})
        finally:
            await close_model(policy)


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
                 count: Annotated[int, Field(ge=1, lt=10)] = 3) -> str:
    """Search Wiki-2018 for a query and return passages with source URLs.

    count is the number of results (1–9, default 3). Each passage is limited to
    1500 characters. Use access(url, info_to_extract) to extract relevant facts.
    """
    docs = await request_tool("/retrieve", {"queries": [query], "topk": count}, runtime)
    result = "\n\n".join(
        f"[Doc {i+1}]({d['url']}):\n{d['contents'][:1500]}"
        for i, d in enumerate(docs)) or "No search results are found."
    return f"# Search query:\n{query}\n# Result:\n{result}"


@tool
async def access(url: str, info_to_extract: str, runtime: ToolRuntime[Context]) -> str:
    """Open a Wiki-2018 page and extract information matching your intent.

    url must be a complete URL returned by search. info_to_extract is the
    specific question or information to extract. Qwen summarizes only the first
    6000 characters; facts absent from this excerpt may exist later in the page.
    """
    page = await request_tool("/access", {"urls": [url]}, runtime)
    content = (page.get("contents") or "") if page else ""
    summary = (await summarize_access(info_to_extract, content[:ACCESS_CHARACTERS], runtime)
               if content.strip() else "No useful Information is Found under this URL.")
    return f"# Access URL:\n{url}\n# Info to extract:\n{info_to_extract}\n# Result:\n{summary}"


class SearchQuery(BaseModel):
    query: str = Field(description="Question to be searched.")
    count: int = Field(default=3, ge=1, lt=10, description="Number of results, less than 10.")


class AccessURL(BaseModel):
    url: str = Field(description="Complete URL returned by the search tool.")
    info_to_extract: str = Field(description="Specific question or information to extract from this URL.")


@tool("search")
async def search_batch(queries: Annotated[list[SearchQuery], Field(min_length=1, max_length=5)],
                       runtime: ToolRuntime[Context]) -> str:
    """Search Wiki-2018 for up to five queries concurrently, returning passages and URLs.

    Each query has a count (1–9, default 3). Each passage is limited to 1500
    characters. Use access with info_to_extract for relevant facts from a page.
    """
    results = await asyncio.gather(*(search.coroutine(q.query, runtime, count=q.count) for q in queries))
    return "\n\n".join(results)


@tool("access")
async def access_batch(urls: Annotated[list[AccessURL], Field(min_length=1, max_length=5)],
                       runtime: ToolRuntime[Context]) -> str:
    """Open up to five Wiki-2018 URLs concurrently and extract facts matching your intent.

    Each item needs a complete url returned by search and info_to_extract describing
    the specific question or information to extract. Qwen summarizes only the first
    6000 characters of each page; later facts may be absent from the result.
    """
    results = await asyncio.gather(*(access.coroutine(u.url, u.info_to_extract, runtime) for u in urls))
    return "\n\n".join(results)
