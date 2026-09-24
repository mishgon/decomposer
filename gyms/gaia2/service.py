"""Python 3.12 Decomposer episode service for GAIA2."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, message_to_dict
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from decomposer.chat_vllm import ChatVLLM
from decomposer.core import TERMINAL_STATUSES, create_decomposer_agent
from decomposer.prompt_profiles import system_prompt_middleware
from gyms.gaia2.model_overflow import (
    ExactModelCallLimitMiddleware,
    Gaia2ModelOverflowError,
    Gaia2ModelOverflowMiddleware,
)
from gyms.gaia2.prompts import compose_decomposer_system_prompt


# ARE's own user-role templates (external/gaia2: agents/default_agent/base_agent.py
# DEFAULT_STEP_2_MESSAGE "task" and "environment_notifications", and
# steps/are_simulation.py format_notification). Pinned by tests/gaia2/test_service.py.
ARE_TASK_TEMPLATE = "[TASK]: \n{content}\n"
ARE_ENVIRONMENT_NOTIFICATIONS_TEMPLATE = (
    "Environment notifications updates:\n***\n{content}\n***\n"
)
ARE_NOTIFICATION_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class EpisodeContext(TypedDict):
    tool_schemas: list[dict[str, Any]]
    broker_url: str
    session_token: str
    policy: str
    scenario_id: str
    run_number: int | None
    notification_cursor: int
    # ARE's native system prompt for the workers, rendered by the proxy per scenario.
    worker_system_prompt: NotRequired[str]


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
    parallel_tool_calls = value.get("parallel_tool_calls", False)
    if not isinstance(parallel_tool_calls, bool):
        raise ValueError("manager.parallel_tool_calls must be a boolean")
    api_key = value.get("api_key")
    api_key_env = value.get("api_key_env", "OPENAI_API_KEY")
    if api_key is None:
        api_key = os.environ.get(api_key_env, "EMPTY")
    use_responses_api = value.get("use_responses_api", False)
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
        "use_responses_api": use_responses_api,
        "timeout": value.get("timeout", 3300),
        "max_retries": value.get("max_retries", 2),
        "model_kwargs": {"parallel_tool_calls": parallel_tool_calls},
    }
    extra_body = dict(value.get("extra_body") or {})
    known = {key: item for key, item in known.items() if item is not None}
    if extra_body:
        known["extra_body"] = extra_body
    if use_responses_api:
        return ChatOpenAI(**known)
    return ChatVLLM(preserve_reasoning=True, **known)


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


def _visible_message_text(message: AIMessage) -> str:
    """Extract user-visible text without leaking Responses API reasoning blocks."""

    if isinstance(message.content, str):
        return message.content
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, str):
            parts.append(block)
        elif not isinstance(block, Mapping):
            continue
        elif block.get("type") in {"text", "output_text"} and isinstance(
            block.get("text"), str
        ):
            parts.append(block["text"])
        elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
            parts.append(block["refusal"])
    return "".join(parts)


def _decomposer_system_prompt(config: dict[str, Any]) -> str:
    profile = config.get("decomposer_system_prompt_profile", "student")
    addendum_profile = config.get("decomposer_system_prompt_addendum_profile")
    return compose_decomposer_system_prompt(profile, addendum_profile)


def _public_context(context: EpisodeContext) -> dict[str, Any]:
    public = {
        "tool_schemas": context["tool_schemas"],
        "broker_url": context["broker_url"],
        "session_token": "<redacted>",
        "policy": context["policy"],
        "scenario_id": context["scenario_id"],
        "run_number": context["run_number"],
        "notification_cursor": context["notification_cursor"],
    }
    worker_system_prompt = context.get("worker_system_prompt")
    if worker_system_prompt is not None:
        public["worker_system_prompt_sha256"] = hashlib.sha256(
            worker_system_prompt.encode("utf-8")
        ).hexdigest()
    return public


def _subagent_summary(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """One entry per subagent run, and the runs whose response the manager never received.

    Subagents persist across runs (and turns), so the type lives on the subagent,
    not on the run.
    """
    subagents = state.get("subagents") or {}
    summaries: list[dict[str, Any]] = []
    outstanding: list[str] = []
    for run_id, run in (state.get("subagent_runs") or {}).items():
        subagent = subagents.get(run.get("subagent_id")) or {}
        item = {
            "subagent_run_id": run_id,
            "subagent_id": run.get("subagent_id"),
            "subagent_type_id": subagent.get("subagent_type_id"),
            "status": run.get("status"),
            "prompt": run.get("prompt"),
            "response": run.get("response"),
            "error": run.get("error"),
            "tool_calls": run.get("tool_calls", []),
            "response_sequence_number": run.get("response_sequence_number"),
        }
        summaries.append(item)
        if (
            run.get("status") not in TERMINAL_STATUSES
            or run.get("response_sequence_number") is None
        ):
            outstanding.append(run_id)
    return summaries, outstanding


def _turn_messages(request: TurnRequest) -> list[dict[str, str]]:
    """The manager's input for one turn: the user-role text ARE gives its own agent.

    ARE's agent runs `run(task=...)` on every wake-up, logging the new user
    messages as one task (`are_simulation_main.py:389-414, 482-486`), and injects
    environment notifications as a separate user-role update (`steps/are_simulation.py`).
    Nothing else is added, so the Decomposer and the native baseline read the same
    turn. Other notification types (e.g. the environment stop) are not rendered.
    """
    user_messages = [
        str(notification.get("message", ""))
        for notification in request.notifications
        if notification.get("type") == "USER_MESSAGE"
    ]
    # ARE hands attachments to its agent as images; the manager gets them as
    # JSON text. Execution and Search user messages carry none.
    attachments = [
        attachment
        for notification in request.notifications
        if notification.get("type") == "USER_MESSAGE"
        for attachment in notification.get("attachments") or []
    ]
    content = "\n".join(user_messages)
    if attachments:
        content += "\nAttachments: " + json.dumps(attachments, ensure_ascii=False)
    messages = [{"role": "user", "content": ARE_TASK_TEMPLATE.format(content=content)}]

    environment_notifications = [
        "["
        + datetime.fromisoformat(notification["simulated_timestamp"]).strftime(
            ARE_NOTIFICATION_TIME_FORMAT
        )
        + "] "
        + str(notification.get("message", ""))
        for notification in request.notifications
        if notification.get("type") == "ENVIRONMENT_NOTIFICATION"
    ]
    if environment_notifications:
        messages.append(
            {
                "role": "user",
                "content": ARE_ENVIRONMENT_NOTIFICATIONS_TEMPLATE.format(
                    content="\n".join(environment_notifications)
                ),
            }
        )
    return messages


def create_app(config: dict[str, Any]) -> FastAPI:
    manager_model = _model_from_config(dict(config.get("manager") or {}))
    subagent_types = config.get("subagent_types") or []
    if not subagent_types:
        raise ValueError("At least one subagent_types entry must be configured")
    middleware = [
        system_prompt_middleware(_decomposer_system_prompt(config)),
        Gaia2ModelOverflowMiddleware(
            "manager",
            max_completion_tokens=config["manager"].get("max_completion_tokens"),
            max_model_len=config.get("max_model_len"),
        )
    ]
    manager_max_model_calls = config.get("manager_max_model_calls")
    if manager_max_model_calls is not None:
        manager_max_model_calls = int(manager_max_model_calls)
        if manager_max_model_calls < 1:
            raise ValueError("manager_max_model_calls must be at least 1")
        middleware.append(
            ExactModelCallLimitMiddleware(
                run_limit=manager_max_model_calls,
                exit_behavior="end",
            )
        )
    checkpointer = InMemorySaver()
    graph = create_decomposer_agent(
        decomposer_model=manager_model,
        subagent_types=subagent_types,
        checkpointer=checkpointer,
        context_schema=EpisodeContext,
        middleware=middleware,
        subagent_recursion_limit=int(config.get("subagent_recursion_limit", 200)),
    )
    recursion_limit = int(config.get("manager_recursion_limit", 200))
    episodes: dict[str, Episode] = {}
    lock = asyncio.Lock()
    app = FastAPI(title="GAIA2 Decomposer Service", version="1")

    def _subagent_client(subagent: dict[str, Any]) -> Any:
        subagent_type = next(
            (
                item
                for item in subagent_types
                if item["subagent_type_id"] == subagent.get("subagent_type_id")
            ),
            None,
        )
        if subagent_type is None or not subagent_type.get("url"):
            return None
        from langgraph_sdk import get_client

        return get_client(url=subagent_type["url"], headers=subagent_type.get("headers"))

    async def cancel_subagents(episode: Episode, *, delete_threads: bool = False) -> None:
        """Cancel the episode's active subagent runs; optionally delete their threads.

        Subagents persist across turns, so their threads are deleted only when the
        episode itself ends.
        """
        try:
            snapshot = await graph.aget_state(
                {"configurable": {"thread_id": episode.thread_id}}
            )
            state = snapshot.values
        except Exception:
            return
        subagents = state.get("subagents") or {}
        for run in (state.get("subagent_runs") or {}).values():
            if run.get("status") in TERMINAL_STATUSES:
                continue
            subagent = subagents.get(run.get("subagent_id")) or {}
            client = _subagent_client(subagent)
            if client is None or not subagent.get("thread_id"):
                continue
            try:
                await client.runs.cancel(
                    thread_id=subagent["thread_id"], run_id=run["run_id"], wait=False
                )
            except Exception:
                pass
        if not delete_threads:
            return
        for subagent in subagents.values():
            client = _subagent_client(subagent)
            if client is None or not subagent.get("thread_id"):
                continue
            try:
                await client.threads.delete(thread_id=subagent["thread_id"])
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
        input_value = {"messages": _turn_messages(request)}
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
        except Gaia2ModelOverflowError as error:
            await cancel_subagents(episode)
            result = {
                "failure": error.as_dict(),
                "timing": {"elapsed_seconds": time.monotonic() - before},
                "trace": {
                    "runtime_context": _public_context(episode.context),
                    "notification_cursor": request.notification_cursor,
                    "turn_number": request.turn_number,
                },
            }
            episode.turns.append(result)
            return result
        finally:
            episode.task = None
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            raise HTTPException(502, "Decomposer produced no final AI message")
        final_text = _visible_message_text(messages[-1])
        if not final_text.strip():
            raise HTTPException(502, "Decomposer produced no visible final text")
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
        await cancel_subagents(episode, delete_threads=True)
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
