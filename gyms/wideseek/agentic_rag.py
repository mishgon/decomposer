"""The monorepo's cited-search researcher harness over its hosted Search API.

Prompts, tool descriptions and settings follow the monorepo at 1a73e8a:
common/tools/agentic_rag.py, common/toolsets/agentic_rag_search.py, common/tasks/agentic_rag.py
and configs/filter_difficulty/agentic_rag.toml. A run allows 32 model calls, with the last
reserved for finish. The gyms impose no per-run wall-clock budget.
"""
import asyncio
import json
import inspect
import os
import time
from typing import Annotated, NotRequired
from functools import lru_cache
from uuid import uuid4

import httpx
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import ToolNode
from langgraph.config import get_config
from jsonschema import Draft202012Validator

from decomposer.run_budget import RunBudgetMiddleware, RunBudgetState
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
SOURCE = "wikipedia"


def search_api():
    """The Search API front and the headers that authorize a request to it."""
    return (os.environ.get("AGENTIC_RAG_URL", "https://lmrouter.2a2i.org/rag").rstrip("/"),
            {"Authorization": "Bearer " + os.environ["AGENTIC_RAG_API_KEY"]})


def _transient(error):
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, httpx.HTTPStatusError) and (
        error.response.status_code in {408, 409, 429} or error.response.status_code >= 500)


# As in the monorepo, transient errors are waited out; the episode deadline bounds the wait.
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


TOOLS = [find, finish, open_page, search]
for tool in TOOLS:
    tool.description = inspect.cleandoc(tool.description)

# Match MCP 2.0 schemas and the monorepo's canonical_tools, including title fields.
TOOL_SCHEMAS = []
for tool in TOOLS:
    parameters = tool.tool_call_schema.model_json_schema()
    parameters.pop("description", None)
    parameters["title"] = tool.name + "Arguments"
    TOOL_SCHEMAS.append({"type": "function", "function": {
        "name": tool.name, "description": tool.description, "parameters": parameters}})
TOOL_SCHEMAS = json.loads(json.dumps(TOOL_SCHEMAS, sort_keys=True, ensure_ascii=False))


def model_messages(messages):
    # Reports adapt the tool result to Decomposer's interface, not the researcher conversation.
    return [message.model_copy(update={"name": None}) if isinstance(message, ToolMessage) else message
            for message in messages if not message.additional_kwargs.get("decomposer_report")]


def report_message(content=""):
    return AIMessage(content=content, additional_kwargs={"decomposer_report": True})


def _merge(pages, update):
    return {**pages, **update}


INVALID_TURN = "No valid tool was called. Check the tool name and argument format, then call a tool correctly. If you want to finish, call finish with all required arguments."
FINISH_REQUEST = "Your work budget has expired. Call finish using the evidence already collected. No other tools are allowed. Follow the answer and source requirements in the system instructions and the finish tool description."
MAX_TURNS = 32
LAST_TURN_REQUEST = "This is your last model call. Call finish exactly once using the evidence already collected. No other tools are allowed. Follow the answer and source requirements in the system instructions and the finish tool description."


@lru_cache
def context_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")


def prompt_tokens(messages, tools):
    wire = convert_to_openai_messages([SystemMessage(SYSTEM_PROMPT), *model_messages(messages)])
    for message in wire:
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                try:
                    call["function"]["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    call["function"]["arguments"] = {"invalid_arguments": arguments}
    return len(context_tokenizer().apply_chat_template(
        wire, tools=[convert_to_openai_tool(tool) for tool in tools],
        tokenize=True, add_generation_prompt=True, enable_thinking=False))


class SearchState(RunBudgetState):
    completed_turns: NotRequired[int]
    pages: NotRequired[Annotated[dict[str, int], _merge]]
    output_tokens: NotRequired[int]
    invalid_turns: NotRequired[int]
    previous_call: NotRequired[dict | None]
    repeated_calls: NotRequired[int]
    end_status: NotRequired[str | None]


class ResearchProtocol(RunBudgetMiddleware):
    """Execute the original tool protocol with model-turn and wall-clock limits."""
    state_schema = SearchState

    def __init__(self, tool_middleware=()):
        self.tool_middleware = tuple(middleware for middleware in tool_middleware
                                    if (type(middleware).awrap_tool_call is not AgentMiddleware.awrap_tool_call
                                        or type(middleware).wrap_tool_call is not AgentMiddleware.wrap_tool_call))
        self.tool_node = ToolNode(TOOLS, awrap_tool_call=self.awrap_tool_call)

    async def awrap_tool_call(self, request, handler):
        # ToolNode executes one call at a time; retain create_agent's wrapper order.
        for middleware in reversed(self.tool_middleware):
            inner = handler

            async def wrapped(request, middleware=middleware, inner=inner):
                return await middleware.awrap_tool_call(request, inner)

            handler = wrapped
        return await super().awrap_tool_call(request, handler)

    def before_agent(self, state, runtime):
        return {**super().before_agent(state, runtime), "invalid_turns": 0, "output_tokens": 0,
                "previous_call": None, "repeated_calls": 0, "end_status": None, "completed_turns": 0}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if state["completed_turns"] >= MAX_TURNS:
            return {"end_status": "turn_limit", "messages": [report_message()], "jump_to": "end"}
        update = super().before_model(state, runtime)
        if update:
            update["messages"] = [HumanMessage(FINISH_REQUEST)]
        if state["completed_turns"] == MAX_TURNS - 1:
            update = {"shutdown_requested": True, "messages": [HumanMessage(LAST_TURN_REQUEST)]}
        messages = [*state["messages"], *(update or {}).get("messages", [])]
        status = ("output_limit" if state["output_tokens"] >= 65536 else
                  "context_limit" if 65536 - prompt_tokens(messages, TOOL_SCHEMAS) - 512 < 4096 else None)
        if status:
            return {"end_status": status, "messages": [report_message()], "jump_to": "end"}
        return update

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        return await asyncio.to_thread(self.before_model, state, runtime)

    def _prepare_model_request(self, request):
        return request.override(messages=model_messages(request.messages), tools=TOOL_SCHEMAS,
                                model_settings={**request.model_settings, "max_tokens": 4096})

    def after_model(self, state, runtime):
        raise NotImplementedError("ResearchProtocol requires async execution (ainvoke/astream).")

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(self, state, runtime):
        message = state["messages"][-1]
        updates = {"messages": [], "pages": {}, "previous_call": state["previous_call"],
                   "completed_turns": state["completed_turns"] + 1,
                   "repeated_calls": state["repeated_calls"],
                   "output_tokens": state["output_tokens"] + (message.usage_metadata or {}).get("output_tokens", 0)}
        valid = 0
        validators = {schema["function"]["name"]: Draft202012Validator(schema["function"]["parameters"])
                      for schema in TOOL_SCHEMAS}
        calls = [*message.tool_calls, *[{**call, "args": None} for call in message.invalid_tool_calls]]
        if raw_calls := message.additional_kwargs.get("tool_calls"):
            # LangChain separates malformed calls; preserve the provider's original batch order.
            by_id = {call["id"]: call for call in calls}
            calls = [by_id[call["id"]] for call in raw_calls]
        for call in calls:
            name, args = call["name"], call["args"]
            errors = ([{"path": [], "message": "Unknown tool name."}] if name not in validators else
                      [{"path": [], "message": "Arguments must be a JSON object."}] if not isinstance(args, dict) else
                      [{"path": list(e.path), "message": e.message} for e in validators[name].iter_errors(args)])
            if errors:
                result = {"error": "invalid_tool_call_format", "details": errors}
            elif state["shutdown_requested"] and name != "finish":
                valid += 1
                result = {"error": "finish_required", "message": "Only finish is allowed on the final attempt."}
            else:
                valid += 1
                key = {"name": name, "arguments": args}
                updates["repeated_calls"] = updates["repeated_calls"] + 1 if key == updates["previous_call"] else 1
                updates["previous_call"] = key
                if updates["repeated_calls"] >= 3 and not state["shutdown_requested"]:
                    result = {"error": "loop_detected", "rejected": True}
                    updates["end_status"] = "loop_detected"
                else:
                    timeout_at = state["timeout_at"]
                    remaining = None if timeout_at is None else max(0, timeout_at - time.monotonic())
                    tool_state = {**state, "pages": {**state.get("pages", {}), **updates["pages"]},
                                  "messages": [message.model_copy(update={"tool_calls": [call]})]}
                    async with asyncio.timeout(remaining):
                        output = await self.tool_node.ainvoke(tool_state, get_config(), runtime=runtime)
                    for item in output if isinstance(output, list) else [output]:
                        update = item.update if isinstance(item, Command) else item
                        updates["messages"].extend(update.get("messages", []))
                        updates["pages"].update(update.get("pages", {}))
                    tool_message = updates["messages"][-1]
                    if name == "finish" and tool_message.artifact:
                        updates["end_status"] = "finished"
                        updates["messages"].append(report_message(tool_message.artifact))
                    if not updates.get("end_status"):
                        continue
                    break
            updates["messages"].append(ToolMessage(json.dumps(result), name=name, tool_call_id=call["id"]))
            if updates.get("end_status"):
                break
        updates["invalid_turns"] = 0 if valid else state["invalid_turns"] + 1
        if not valid:
            if updates["invalid_turns"] >= 3:
                updates["end_status"] = "invalid_tool_call"
            else:
                updates["messages"].append(HumanMessage(INVALID_TURN))
        if updates.get("end_status"):
            report = updates["messages"].pop() if updates["end_status"] == "finished" else report_message()
            completed = {m.tool_call_id for m in updates["messages"] if isinstance(m, ToolMessage)}
            for call in calls:
                if call["id"] not in completed:
                    updates["messages"].append(ToolMessage(
                        json.dumps({"error": "session_ended", "rejected": True}),
                        name=call["name"], tool_call_id=call["id"]))
            updates["messages"].append(report)
            updates["jump_to"] = "end"
        else:
            updates["jump_to"] = "model"
        return updates
