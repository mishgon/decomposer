"""Shared episode budget and model/tool logs; no secrets in artifacts."""
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import TypedDict
from uuid import uuid4

import httpx
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool


GENERATION = {"temperature": .7, "top_p": .8, "presence_penalty": 1.5,
              "extra_body": {"top_k": 20, "min_p": 0.,
              "repetition_penalty": 1., "chat_template_kwargs": {"enable_thinking": False}}}


class Context(TypedDict):
    directory: str


class BudgetExceeded(RuntimeError):
    pass


def directory(context):
    path = Path(context["directory"]).resolve()
    root = Path(os.environ.get("WS_ARTIFACT_ROOT", "artifacts/gyms/wideseek/runs")).resolve()
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


class HostTransport(httpx.AsyncBaseTransport):
    """Connect to a configured IP without disabling hostname TLS verification."""
    def __init__(self):
        self.inner = httpx.AsyncHTTPTransport(retries=1)

    async def handle_async_request(self, request):
        mapping = os.environ.get("WS_MODEL_HOST", "")
        if mapping:
            host, ip = mapping.split(":", 1)
            if request.url.host == host:
                request.extensions["sni_hostname"] = host
                request.headers["Host"] = host
                request.url = request.url.copy_with(host=ip)
        return await self.inner.handle_async_request(request)

    async def aclose(self):
        await self.inner.aclose()


def model(model_id=None):
    return ChatOpenAI(model=model_id or os.environ.get("WS_MODEL", "Qwen/Qwen3.5-4B"),
        base_url=os.environ["LLM_PROXY_URL"], api_key=os.environ["LLM_PROXY_MASTER_KEY"],
        **GENERATION,
        timeout=180, max_retries=2, use_responses_api=False,
        http_async_client=httpx.AsyncClient(transport=HostTransport(), trust_env=False))


class ModelLog(AgentMiddleware):
    def __init__(self, role):
        self.role = role

    async def awrap_model_call(self, request, handler):
        path = await asyncio.to_thread(directory, request.runtime.context)
        limit = await asyncio.to_thread(reserve, path)
        log = path / "model_calls" / f"{uuid4().hex}.json"
        row = {"role": self.role, "started_at": time.time(), "max_tokens": limit,
               "generation": {**GENERATION, **request.model_settings, "max_tokens": limit},
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
