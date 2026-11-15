"""Shared episode budget and model/tool logs; no secrets in artifacts."""
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4
from datetime import datetime, timezone

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, message_to_dict
from langgraph.config import get_config
from gyms.model_logging import append_record, request_delta
from decomposer.models import create_model as model
from decomposer.run_budget import RunBudgetContext


MODEL_BACKEND = os.environ.get("WS_MODEL_BACKEND", "hosted")
if MODEL_BACKEND not in {"hosted", "local"}:
    raise ValueError("WS_MODEL_BACKEND must be hosted or local")
DEFAULT_MODEL = "vllm/qwen_3_5_4b_non_thinking" if MODEL_BACKEND == "local" else "lmrouter/qwen_3_5_4b_thinking"
DEFAULT_SUBAGENT = DEFAULT_MODEL
DEFAULT_TEACHER = "vllm/qwen_3_5_4b_non_thinking" if MODEL_BACKEND == "local" else "lmrouter/qwen_3_8_flash_next_non_thinking"
MODEL_PROFILES = (DEFAULT_MODEL, DEFAULT_SUBAGENT, DEFAULT_TEACHER)


def model_metadata(policy):
    return {name: getattr(policy, name, None) for name in (
        "model_name", "temperature", "top_p", "presence_penalty", "extra_body",
        "max_tokens", "preserve_reasoning", "max_retries")}


async def close_model(policy):
    if client := getattr(policy, "http_client", None):
        client.close()
    if client := getattr(policy, "http_async_client", None):
        await client.aclose()


class Context(RunBudgetContext):
    directory: str


class BudgetExceeded(RuntimeError):
    pass


def directory(context):
    # The runner supplies this path through trusted runtime context, not tool arguments.
    return Path(context["directory"]).resolve()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, default=str, indent=2))
    temporary.replace(path)


def init_budget(path, calls=None, tokens=None, *, coordinated=False):
    path.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(path / "budget.sqlite") as db:
        db.execute("CREATE TABLE budget (calls INTEGER, tokens INTEGER, coordinated INTEGER, closing INTEGER)")
        db.execute("INSERT INTO budget VALUES (?,?,?,0)", (calls, tokens, coordinated and calls is not None))
        db.execute("CREATE TABLE workers (thread_id TEXT PRIMARY KEY)")


def coordinated_budget(path):
    with sqlite3.connect(path / "budget.sqlite") as db:
        return bool(db.execute("SELECT coordinated FROM budget").fetchone()[0])


def register_worker(path, thread_id):
    """Reserve one report per active worker and one coordinator answer atomically."""
    with sqlite3.connect(path / "budget.sqlite", timeout=30) as db:
        db.execute("BEGIN IMMEDIATE")
        calls, tokens, coordinated, closing = db.execute("SELECT * FROM budget").fetchone()
        if not coordinated:
            return True
        active = db.execute("SELECT count(*) FROM workers").fetchone()[0]
        if closing or active >= 3 or calls < active + 2 or (tokens is not None and tokens < (active + 2) * 4096):
            return False
        return db.execute("INSERT OR IGNORE INTO workers VALUES (?)", (thread_id,)).rowcount == 1


def release_worker(path, thread_id):
    with sqlite3.connect(path / "budget.sqlite", timeout=30) as db:
        db.execute("DELETE FROM workers WHERE thread_id=?", (thread_id,))


def reserve(path, *, with_remaining=False, final_response=False, role=None, thread_id=None):
    with sqlite3.connect(path / "budget.sqlite", timeout=30) as db:
        db.execute("BEGIN IMMEDIATE")
        calls, tokens, coordinated, closing = db.execute("SELECT * FROM budget").fetchone()
        if (calls is not None and calls < 1) or (tokens is not None and tokens < 1):
            raise BudgetExceeded("Shared episode model budget exhausted")
        final = final_response and calls == 1
        held_tokens = 0
        if coordinated:
            active = db.execute("SELECT count(*) FROM workers").fetchone()[0]
            if role == "researcher":
                if not db.execute("SELECT 1 FROM workers WHERE thread_id=?", (thread_id,)).fetchone():
                    raise BudgetExceeded("Worker has no reserved report")
                final = bool(closing or calls <= active + 1 or
                             (tokens is not None and tokens <= (active + 1) * 4096))
                held_tokens = (active if final else active + 1) * 4096
                if final:
                    db.execute("UPDATE budget SET closing=1")
                    db.execute("DELETE FROM workers WHERE thread_id=?", (thread_id,))
            else:
                final = bool(closing or calls == 1 or (tokens is not None and tokens <= 4096))
                if active:
                    raise BudgetExceeded("Collect active workers before calling coordinator")
        limit = 16384 if final else (4096 if tokens is not None else None)
        if tokens is not None:
            limit = min(limit, tokens - held_tokens)
        if limit is not None and limit < 1:
            raise BudgetExceeded("No tokens remain outside reserved reports")
        db.execute("UPDATE budget SET calls=calls-1,tokens=tokens-?", (limit,))
    remaining = calls - 1 if calls is not None else None
    return (limit, remaining, final) if role else ((limit, remaining) if with_remaining else limit)


def refund(path, amount):
    with sqlite3.connect(path / "budget.sqlite", timeout=30) as db:
        db.execute("UPDATE budget SET tokens=tokens+?", (amount,))


class ModelLog(AgentMiddleware):
    def __init__(self, role, *, final_response=False):
        self.role = role
        self.final_response = final_response

    async def awrap_model_call(self, request, handler):
        path = await asyncio.to_thread(directory, request.runtime.context)
        thread_id = get_config()["configurable"]["thread_id"]
        coordinated = await asyncio.to_thread(coordinated_budget, path)
        if coordinated and self.role == "decomposer" and any(
            "response_sequence_number" not in run for run in request.state.get("agent_runs", {}).values()
        ):
            # Waiting is control flow, not inference; preserve calls for worker reports.
            message = AIMessage(content="", tool_calls=[{"name": "wait", "args": {}, "id": uuid4().hex}],
                                additional_kwargs={"budget_control": "collect_workers"})
            await asyncio.to_thread(append_record, path / "budget_events.jsonl",
                                   {"event": "collect_workers", "thread_id": thread_id, "time": time.time()})
            return ModelResponse(result=[message])
        limit, remaining, final_response = await asyncio.to_thread(
            reserve, path, final_response=self.final_response, role=self.role, thread_id=thread_id)
        log = path / "model_calls.jsonl"
        started = time.monotonic()
        row = {"call_id": uuid4().hex, "role": self.role,
               "thread_id": get_config()["configurable"]["thread_id"],
               "started_at": datetime.now(timezone.utc).isoformat(), "max_tokens": limit,
               "budget_final_response": final_response,
               "generation": {**model_metadata(request.model), **request.model_settings},
               "request_message_count": len(request.messages),
               "request_delta": [message_to_dict(m) for m in request_delta(request.messages)]}
        if not any(m.type == "ai" for m in request.messages):
            row["tools"] = [convert_to_openai_tool(t) for t in request.tools]
            row["system_message"] = message_to_dict(request.system_message) if request.system_message else None
        await asyncio.to_thread(append_record, log, {**row, "status": "started"})
        try:
            settings = dict(request.model_settings)
            if limit is not None:
                settings["max_tokens"] = limit
            system = request.system_message
            overrides = {}
            if final_response:
                # Reserve the last permitted call for an answer, not another tool request.
                overrides["tool_choice"] = "none"
                overrides["tools"] = []
                settings["temperature"] = 0.
                row["budget_finalization_temperature"] = 0.
                if MODEL_BACKEND == "local":
                    extra = {**(getattr(request.model, "extra_body", None) or {}),
                             **settings.get("extra_body", {})}
                    extra["bad_words"] = [*extra.get("bad_words", []), "<tool_call>", "</tool_call>", "<function="]
                    if coordinated and self.role == "researcher":
                        extra["structured_outputs"] = {"regex": r"[\s\S]{1,4000}"}
                        row["budget_finalization"] = "local_report_grammar"
                    elif any(m.type == "human" and "markdown" in m.text.casefold() for m in request.messages):
                        extra["structured_outputs"] = {"regex": r"```markdown\n\|[\s\S]*```"}
                        row["budget_finalization"] = "local_markdown_grammar"
                    settings["extra_body"] = extra
                notice = ("No tool calls remain in this episode's model-call budget. "
                          "Return the requested final answer now using only available evidence. "
                          "Mark unresolved entries as unknown; do not invent missing information. "
                          "Tools are unavailable. Do not emit tool-call markup or instructions to call tools. "
                          "Follow the requested output format exactly, including a markdown code fence "
                          "when requested; do not add commentary outside the requested answer.")
                if coordinated and self.role == "researcher":
                    notice += (" This is a worker report, not the coordinator's final answer. "
                               "In at most 4000 characters, summarize verified findings, source URLs, "
                               "and unresolved gaps. Return partial findings even if the subtask is unfinished.")
                content = system.content if system else ""
                blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
                system = SystemMessage(content=[*blocks, {"type": "text", "text": notice}])
            response = await handler(request.override(model_settings=settings, system_message=system, **overrides))
            outcome = {"status": "success", "response": [message_to_dict(m) for m in response.result]}
            used = sum((getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0)
                       for m in response.result)
            # If provider omits usage, conservatively keep the full reservation.
            if limit is not None and any(getattr(m, "usage_metadata", None) for m in response.result):
                await asyncio.to_thread(refund, path, max(0, limit - used))
            if coordinated and self.role == "researcher" and not any(m.tool_calls for m in response.result):
                await asyncio.to_thread(release_worker, path, thread_id)
            return response
        except BaseException as exc:
            outcome = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            if coordinated and self.role == "researcher":
                await asyncio.to_thread(release_worker, path, thread_id)
            raise
        finally:
            await asyncio.to_thread(append_record, log, {
                "call_id": row["call_id"], "role": self.role, **outcome,
                "generation": {**model_metadata(request.model), **settings},
                "effective_tool_count": len(overrides.get("tools", request.tools)),
                **{key: row[key] for key in ("budget_finalization", "budget_finalization_temperature") if key in row},
                **({"effective_system_message": message_to_dict(system)} if overrides else {}),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - started})


    async def awrap_tool_call(self, request, handler):
        if self.role != "decomposer":
            return await handler(request)
        path = directory(request.runtime.context)
        if request.tool_call["name"] == "wait":
            result = await handler(request)
            if hasattr(result, "update") and isinstance(result.update, dict):
                for run in result.update.get("agent_runs", {}).values():
                    if "response_sequence_number" in run:
                        agent = request.state["agents"][run["agent_id"]]
                        await asyncio.to_thread(release_worker, path, agent["thread_id"])
            return result
        if request.tool_call["name"] != "run":
            return await handler(request)
        agent_id = request.tool_call["args"].get("agent_id")
        agent = request.state.get("agents", {}).get(agent_id)
        if agent is None:
            return await handler(request)
        thread_id = agent["thread_id"]
        if not await asyncio.to_thread(register_worker, path, thread_id):
            return ToolMessage(content="No research budget or worker slot remains. Collect existing reports and return available evidence; mark missing facts unknown.",
                               tool_call_id=request.tool_call["id"], name="run", status="error")
        try:
            result = await handler(request)
        except BaseException:
            await asyncio.to_thread(release_worker, path, thread_id)
            raise
        if not hasattr(result, "update") or not isinstance(result.update, dict) or not result.update.get("agent_runs"):
            await asyncio.to_thread(release_worker, path, thread_id)
        return result
