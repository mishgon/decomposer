"""The monorepo's cited-search researcher harness over its hosted Search API.

Prompts, tool descriptions and settings follow the monorepo at 1a73e8a:
common/tools/agentic_rag.py, common/toolsets/agentic_rag_search.py, common/harness/agentic_rag.py,
common/tasks/agentic_rag.py and configs/filter_difficulty/agentic_rag.toml.
"""
import asyncio
import json
import os
import time
from typing import Annotated, NotRequired
from uuid import uuid4

import httpx
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import Field
from tenacity import retry, retry_if_exception, wait_exponential_jitter

from gyms.wideseek.runtime import Context, directory, save

SYSTEM_PROMPT = """Answer using retrieved evidence only. Finish with your answer and supporting sources. Do not list competing answers unless requested.

When instructed to finish, submit the best answer supported by the evidence already inspected, even if further verification is unfinished. If the evidence does not support an answer, briefly explain what remains unknown. Even if you cannot answer the question, include relevant sources that support a partial answer or intermediate findings.
Use search to discover pages, find to locate matching passages, and open for context. Inspect evidence with open or find before answering; do not rely on snippets alone. You may call tools in parallel when their inputs are already known. Wait for results before making dependent calls. Avoid repeated or paraphrased searches; change approach when needed. Stop once the evidence is sufficient."""
URL_NOT_IN_SESSION = json.dumps({
    "success": False, "error": "url_not_in_session",
    "message": "URL not found in this session's search results. "
               "Copy a returned URL exactly for open/find; trailing path slashes are handled automatically."})
# The monorepo harness (common/harness/agentic_rag.py) allows 32 model calls; the last may only call finish.
MAX_TURNS = 32
FINAL_ATTEMPT = ("This is your last model call. Call finish exactly once using the evidence "
                 "already collected. No other tools are allowed. Follow the answer and source "
                 "requirements in the system instructions and the finish tool description.")
SOURCE = "wikipedia"


def search_api():
    """The Search API front and the headers that authorize a request to it."""
    return (os.environ.get("AGENTIC_RAG_URL", "https://lmrouter.2a2i.org/rag"),
            {"Authorization": "Bearer " + os.environ["AGENTIC_RAG_API_KEY"]})


def _transient(error):
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, httpx.HTTPStatusError) and (
        error.response.status_code in {408, 409, 429} or error.response.status_code >= 500)


# As in the monorepo, transient errors are waited out; the run budget bounds the wait.
@retry(retry=retry_if_exception(_transient), wait=wait_exponential_jitter(multiplier=0.5, max=30), reraise=True)
async def _post(method, payload):
    url, headers = search_api()
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        response = await client.post(url + "/" + method, headers=headers, json={**payload, "source": SOURCE})
        response.raise_for_status()
        return response.json()


async def request_search_api(method, payload, runtime):
    row = {"tool": method, "arguments": payload, "started_at": time.time()}
    path = (await asyncio.to_thread(directory, runtime.context)) / "tool_calls" / f"{uuid4().hex}.json"
    try:
        row["response"] = await _post(method, payload)
        return row["response"]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        row["finished_at"] = time.time()
        await asyncio.to_thread(save, path, row)


def page_url_key(url):
    """Ignore trailing path slashes while preserving query and fragment verbatim."""
    prefix, fragment_separator, fragment = url.partition("#")
    path, query_separator, query = prefix.partition("?")
    return path.rstrip("/") + query_separator + query + fragment_separator + fragment


def page_url(pages, url):
    """Resolve a submitted URL to its session URL, ignoring trailing slashes."""
    if url in pages:
        return url
    return next((session_url for session_url in pages if page_url_key(session_url) == page_url_key(url)), None)


@tool
async def search(query: Annotated[str, Field(description="Search query.")],
                 runtime: ToolRuntime[Context]) -> Command:
    """Search the corpus. Returns up to 5 relevant pages with URLs, titles, and snippets.

    Use open or find to inspect the supporting evidence.
    """
    hits = (await request_search_api("search", {"query": query, "limit": 5}, runtime))["results"]
    content = json.dumps([{key: hit[key] for key in ("url", "title", "snippet")} for hit in hits],
                         ensure_ascii=False)
    return Command(update={"pages": {hit["url"]: hit["page_id"] for hit in hits},
                           "messages": [ToolMessage(content, name="search", tool_call_id=runtime.tool_call_id)]})


@tool("open")
async def open_page(url: Annotated[str, Field(description="Page URL returned by search in this session.")],
                    runtime: ToolRuntime[Context],
                    loc: Annotated[int | None, Field(
                        description="Zero-based starting line. Omit or use null to start at line 0.")] = None) -> str:
    """Read up to 16 lines from a page returned by search, starting at loc.

    Choose any starting line to inspect a passage or continue reading.
    """
    pages = runtime.state.get("pages", {})
    url = page_url(pages, url)
    if url is None:
        return URL_NOT_IN_SESSION
    result = await request_search_api(
        "open", {"page_id": pages[url], "loc": 0 if loc is None else loc, "num_lines": 16}, runtime)
    return json.dumps(result, ensure_ascii=False)


@tool
async def find(url: Annotated[str, Field(description="Page URL returned by search in this session.")],
               pattern: Annotated[str, Field(description=(
                   "Regular expression matched within each line, case-insensitive. "
                   r"Plain text matches substrings; use A|B for alternatives and \bword\b "
                   r"for whole words. Escape metacharacters for literal matching, e.g. 3\.14. "
                   "Do not wrap the pattern in /.../."))],
               runtime: ToolRuntime[Context]) -> str:
    """Find matching lines in a page returned by search.

    Returns up to 16 matching lines with line numbers and excerpts.
    Use open for surrounding context. Invalid or timed-out patterns return
    feedback; simplify or correct the pattern and retry.
    """
    pages = runtime.state.get("pages", {})
    url = page_url(pages, url)
    if url is None:
        return URL_NOT_IN_SESSION
    result = await request_search_api(
        "find", {"page_id": pages[url], "pattern": pattern, "max_entries": 16}, runtime)
    return json.dumps(result, ensure_ascii=False)


@tool(response_format="content_and_artifact")
def finish(answer: Annotated[str, Field(description=(
               "Direct, consistent answer supported by retrieved evidence. If evidence is insufficient, "
               "explain what remains unknown."))],
           sources: Annotated[list[str], Field(description=(
               "URLs of pages whose content supports the answer, including intermediate "
               "facts needed to derive it. Rank by relevance, include each URL once, "
               "and exclude pages that did not contribute supporting evidence. "
               "Use only URLs returned by retrieval tools in this session. "
               "Use an empty list if no supporting evidence was found."))],
           runtime: ToolRuntime[Context]) -> tuple[str, str | None]:
    """Submit the final answer and its supporting evidence.

    A valid submission ends the session; invalid submissions return feedback
    for correction.
    """
    pages = runtime.state.get("pages", {})
    resolved = [page_url(pages, url) for url in sources]
    unknown = [url for url, match in zip(sources, resolved) if match is None]
    cited = [match for match in resolved if match is not None]
    checks = [
        (bool(answer.strip()), "answer", "Provide a nonempty answer or explain why evidence is insufficient.", None),
        (len(cited) == len(set(cited)), "sources", "Include each source URL only once.", None),
        (not unknown, "sources", "Use only source URLs returned in this session.", unknown),
    ]
    details = [{"field": field, "message": message, **({"urls": urls} if urls else {})}
               for valid, field, message, urls in checks if not valid]
    if details:
        return json.dumps({"success": False, "error": "invalid_answer", "details": details}), None
    # The artifact is the report Decomposer reads as the run's final AI message.
    return json.dumps({"success": True}), "\n\n".join([answer, *(["Sources:\n" + "\n".join(cited)] if cited else [])])


TOOLS = [search, open_page, find, finish]


def _merge(pages, update):
    return {**pages, **update}


class SearchState(AgentState):
    pages: NotRequired[Annotated[dict[str, int], _merge]]


class FinishReport(AgentMiddleware):
    """Keep the session's searched pages and end the run on a valid finish, its report last."""
    state_schema = SearchState

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        for message in reversed(state["messages"]):
            if not isinstance(message, ToolMessage):
                return None
            if message.name == "finish" and message.artifact:
                return {"messages": [AIMessage(content=message.artifact)], "jump_to": "end"}
        return None


class TurnState(AgentState):
    turns: NotRequired[int]


class TurnLimit(AgentMiddleware):
    """Allow MAX_TURNS model calls per run; the last may only call finish."""
    state_schema = TurnState

    def before_agent(self, state, runtime):
        return {"turns": 0}

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if state["turns"] == MAX_TURNS - 1:
            return {"messages": [HumanMessage(content=FINAL_ATTEMPT)]}
        if state["turns"] >= MAX_TURNS:
            # The last call's finish was invalid; its answer is the report.
            call = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage)).tool_calls
            answer = next((c["args"].get("answer", "") for c in call if c["name"] == "finish"), "")
            return {"messages": [AIMessage(content=answer)], "jump_to": "end"}
        return None

    def after_model(self, state, runtime):
        return {"turns": state["turns"] + 1}

    async def awrap_model_call(self, request, handler):
        if request.state["turns"] == MAX_TURNS - 1:
            request = request.override(tools=[tool for tool in request.tools if tool.name == "finish"])
        return await handler(request)
