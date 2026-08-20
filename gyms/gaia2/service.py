"""Python 3.12 Decomposer episode service for GAIA2."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, message_to_dict
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from decomposer.core import TERMINAL_STATUSES, create_decomposer_agent


class EpisodeContext(TypedDict):
    tool_schemas: list[dict[str, Any]]
    broker_url: str
    session_token: str
    policy: str
    scenario_id: str
    run_number: int | None
    notification_cursor: int


class CreateEpisodeRequest(BaseModel):
    context: EpisodeContext


class TurnRequest(BaseModel):
    notifications: list[dict[str, Any]] = Field(min_length=1)
    notification_cursor: int
    turn_number: int


@dataclass
class Episode:
    episode_id: str
    thread_id: str
    context: EpisodeContext
    created_at: float = field(default_factory=time.time)
    turns: list[dict[str, Any]] = field(default_factory=list)
    task: asyncio.Task | None = None
    cancelled: bool = False


def _model_from_config(value: dict[str, Any]) -> ChatOpenAI:
    if not value.get("model"):
        raise ValueError("manager.model must be configured")
    api_key = value.get("api_key")
    api_key_env = value.get("api_key_env", "OPENAI_API_KEY")
    if api_key is None:
        api_key = os.environ.get(api_key_env, "EMPTY")
    known = {
        "model": value["model"],
        "base_url": value.get("base_url"),
        "api_key": api_key,
        "temperature": value.get("temperature"),
        "top_p": value.get("top_p"),
        "presence_penalty": value.get("presence_penalty"),
        "max_completion_tokens": value.get("max_completion_tokens"),
        "reasoning_effort": value.get("reasoning_effort"),
        "reasoning": value.get("reasoning"),
        "use_responses_api": value.get("use_responses_api", False),
        "timeout": value.get("timeout", 3300),
        "max_retries": value.get("max_retries", 2),
    }
    extra_body = dict(value.get("extra_body") or {})
    known = {key: item for key, item in known.items() if item is not None}
    if extra_body:
        known["extra_body"] = extra_body
    return ChatOpenAI(**known)


def _usage(messages: list[Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for message in messages:
        metadata = getattr(message, "usage_metadata", None) or {}
        for key, value in metadata.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def _safe_message(message: Any) -> dict[str, Any]:
    value = message_to_dict(message)
    return json.loads(json.dumps(value, default=str))


def _public_context(context: EpisodeContext) -> dict[str, Any]:
    return {
        "tool_schemas": context["tool_schemas"],
        "broker_url": context["broker_url"],
        "session_token": "<redacted>",
        "policy": context["policy"],
        "scenario_id": context["scenario_id"],
        "run_number": context["run_number"],
        "notification_cursor": context["notification_cursor"],
    }


def _subagent_summary(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    summaries: list[dict[str, Any]] = []
    outstanding: list[str] = []
    for run_id, run in (state.get("subagent_runs") or {}).items():
        item = {
            "subagent_run_id": run_id,
            "subagent_type_id": run.get("subagent_type_id"),
            "status": run.get("status"),
            "prompt": run.get("prompt"),
            "report": run.get("report"),
            "tool_calls": run.get("tool_calls", []),
            "report_sequence_number": run.get("report_sequence_number"),
        }
        summaries.append(item)
        if run.get("status") not in TERMINAL_STATUSES or run.get("report") is None:
            outstanding.append(run_id)
    return summaries, outstanding


def _notification_prompt(request: TurnRequest) -> str:
    lines = [
        f"GAIA2 scenario turn {request.turn_number}. New ARE notifications follow."
    ]
    for notification in request.notifications:
        timestamp = notification.get("simulated_timestamp", "unknown time")
        kind = notification.get("type", "notification")
        message = notification.get("message", "")
        lines.append(f"[{timestamp}] {kind}: {message}")
        attachments = notification.get("attachments") or []
        if attachments:
            lines.append("Attachments: " + json.dumps(attachments, ensure_ascii=False))
    lines.append(
        "Delegate the work to subagents. Return exactly the text that should be sent "
        "to the user for this turn. Collect every spawned subagent before answering."
    )
    return "\n\n".join(lines)


def create_app(config: dict[str, Any]) -> FastAPI:
    manager_model = _model_from_config(dict(config.get("manager") or {}))
    subagent_types = config.get("subagent_types") or []
    if not subagent_types:
        raise ValueError("At least one subagent_types entry must be configured")
    checkpointer = InMemorySaver()
    graph = create_decomposer_agent(
        decomposer_model=manager_model,
        subagent_types=subagent_types,
        checkpointer=checkpointer,
        context_schema=EpisodeContext,
        subagent_recursion_limit=int(config.get("subagent_recursion_limit", 200)),
    )
    recursion_limit = int(config.get("manager_recursion_limit", 200))
    episodes: dict[str, Episode] = {}
    lock = asyncio.Lock()
    app = FastAPI(title="GAIA2 Decomposer Service", version="1")

    async def cancel_subagents(episode: Episode) -> None:
        try:
            snapshot = await graph.aget_state(
                {"configurable": {"thread_id": episode.thread_id}}
            )
            state = snapshot.values
        except Exception:
            return
        for run in (state.get("subagent_runs") or {}).values():
            if run.get("status") in TERMINAL_STATUSES:
                continue
            subagent_type = next(
                (
                    item
                    for item in subagent_types
                    if item["subagent_type_id"] == run.get("subagent_type_id")
                ),
                None,
            )
            if subagent_type is None or not subagent_type.get("url"):
                continue
            from langgraph_sdk import get_client

            client = get_client(
                url=subagent_type["url"], headers=subagent_type.get("headers")
            )
            try:
                await client.runs.cancel(
                    thread_id=run["thread_id"], run_id=run["run_id"], wait=False
                )
                await client.threads.delete(thread_id=run["thread_id"])
            except Exception:
                pass

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "episodes": len(episodes)}

    @app.post("/v1/episodes")
    async def create_episode(request: CreateEpisodeRequest) -> dict[str, str]:
        episode_id = uuid.uuid4().hex
        episode = Episode(
            episode_id=episode_id,
            thread_id=f"gaia2-{episode_id}",
            context=request.context,
        )
        async with lock:
            episodes[episode_id] = episode
        return {"episode_id": episode_id}

    @app.post("/v1/episodes/{episode_id}/turn")
    async def invoke_turn(episode_id: str, request: TurnRequest) -> dict[str, Any]:
        episode = episodes.get(episode_id)
        if episode is None:
            raise HTTPException(404, "unknown episode")
        if episode.cancelled:
            raise HTTPException(409, "episode cancelled")
        if episode.task is not None and not episode.task.done():
            raise HTTPException(409, "episode already has an active turn")
        episode.context["notification_cursor"] = request.notification_cursor
        before = time.monotonic()
        input_value = {
            "messages": [{"role": "user", "content": _notification_prompt(request)}]
        }
        episode.task = asyncio.create_task(
            graph.ainvoke(
                input_value,
                config={
                    "configurable": {"thread_id": episode.thread_id},
                    "recursion_limit": recursion_limit,
                },
                context=episode.context,
            )
        )
        try:
            state = await episode.task
        except asyncio.CancelledError:
            await cancel_subagents(episode)
            raise HTTPException(409, "episode cancelled")
        finally:
            episode.task = None
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            raise HTTPException(502, "Decomposer produced no final AI message")
        final_text = messages[-1].content
        if not isinstance(final_text, str):
            final_text = json.dumps(final_text, ensure_ascii=False)
        subagents, outstanding = _subagent_summary(state)
        result = {
            "final_text": final_text,
            "subagent_states": subagents,
            "outstanding_subagents": outstanding,
            "usage": _usage(messages),
            "timing": {"elapsed_seconds": time.monotonic() - before},
            "trace": {
                "manager_messages": [_safe_message(message) for message in messages],
                "runtime_context": _public_context(episode.context),
                "notification_cursor": request.notification_cursor,
                "turn_number": request.turn_number,
            },
        }
        episode.turns.append(result)
        return result

    @app.delete("/v1/episodes/{episode_id}")
    async def delete_episode(episode_id: str) -> dict[str, bool]:
        async with lock:
            episode = episodes.pop(episode_id, None)
        if episode is None:
            return {"deleted": False}
        episode.cancelled = True
        if episode.task is not None and not episode.task.done():
            episode.task.cancel()
            try:
                await episode.task
            except (asyncio.CancelledError, HTTPException):
                pass
        await cancel_subagents(episode)
        checkpointer.delete_thread(episode.thread_id)
        return {"deleted": True}

    return app


def load_config(path: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Service config must be an object")

    def expand(item: Any) -> Any:
        if isinstance(item, str):
            return os.path.expandvars(item)
        if isinstance(item, list):
            return [expand(child) for child in item]
        if isinstance(item, dict):
            return {key: expand(child) for key, child in item.items()}
        return item

    return expand(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    args = parser.parse_args()
    uvicorn.run(create_app(load_config(args.config)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
