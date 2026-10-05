"""Shared episode budget and model/tool logs; no secrets in artifacts."""
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import TypedDict
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.utils.function_calling import convert_to_openai_tool
from decomposer.models import create_model as model


DEFAULT_MODEL = "lmrouter/qwen_3_5_4b_unlooped_non_thinking"
DEFAULT_SUBAGENT = "lmrouter/qwen_3_5_4b_unlooped_thinking"
DEFAULT_TEACHER = "lmrouter/qwen_3_8_flash_next_non_thinking"
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


class Context(TypedDict):
    directory: str


class BudgetExceeded(RuntimeError):
    pass


def directory(context):
    path = Path(context["directory"]).resolve()
    root = Path(os.environ.get("WS_ARTIFACT_ROOT", "artifacts")).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Episode directory must be beneath WS_ARTIFACT_ROOT")
    return path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, default=str, indent=2))
    temporary.replace(path)


def init_budget(path, calls=None, tokens=None):
    path.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(path / "budget.sqlite") as db:
        db.execute("CREATE TABLE budget (calls INTEGER, tokens INTEGER)")
        db.execute("INSERT INTO budget VALUES (?,?)", (calls, tokens))


def reserve(path):
    with sqlite3.connect(path / "budget.sqlite", timeout=30) as db:
        db.execute("BEGIN IMMEDIATE")
        calls, tokens = db.execute("SELECT calls,tokens FROM budget").fetchone()
        if (calls is not None and calls < 1) or (tokens is not None and tokens < 1):
            raise BudgetExceeded("Shared episode model budget exhausted")
        limit = min(tokens, 4096) if tokens is not None else None
        db.execute("UPDATE budget SET calls=calls-1,tokens=tokens-?", (limit,))
    return limit


def refund(path, amount):
    with sqlite3.connect(path / "budget.sqlite", timeout=30) as db:
        db.execute("UPDATE budget SET tokens=tokens+?", (amount,))


class ModelLog(AgentMiddleware):
    def __init__(self, role):
        self.role = role

    async def awrap_model_call(self, request, handler):
        path = await asyncio.to_thread(directory, request.runtime.context)
        limit = await asyncio.to_thread(reserve, path)
        log = path / "model_calls" / f"{uuid4().hex}.json"
        row = {"role": self.role, "started_at": time.time(), "max_tokens": limit,
               "generation": {**model_metadata(request.model), **request.model_settings},
               "tools": [convert_to_openai_tool(t) for t in request.tools],
               "messages": [m.model_dump(mode="json") for m in request.messages]}
        if request.system_message:
            row["system_message"] = request.system_message.model_dump(mode="json")
        await asyncio.to_thread(save, log, row)
        try:
            settings = dict(request.model_settings)
            if limit is not None:
                settings["max_tokens"] = limit
            response = await handler(request.override(model_settings=settings))
            row["responses"] = [m.model_dump(mode="json") for m in response.result]
            used = sum((getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0)
                       for m in response.result)
            # If provider omits usage, conservatively keep the full reservation.
            if limit is not None and any(getattr(m, "usage_metadata", None) for m in response.result):
                await asyncio.to_thread(refund, path, max(0, limit - used))
            return response
        except BaseException as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["finished_at"] = time.time()
            await asyncio.to_thread(save, log, row)
