"""Crash-resistant append-only logging for agent model calls."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone

from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import message_to_dict
from decomposer.model_logging import append_record, request_delta as _request_delta


LOG_PATH_ENV = "TOOLATHLON_AGENT_CALL_LOG"


def _append_record(record: dict) -> None:
    configured = os.environ.get(LOG_PATH_ENV)
    if not configured:
        return
    append_record(configured, record)


@wrap_model_call
async def durable_model_call_log(request, handler):
    configured = os.environ.get(LOG_PATH_ENV)
    if not configured:
        return await handler(request)
    call_id = uuid.uuid4().hex
    started = time.monotonic()
    base = {
        "call_id": call_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": (
            getattr(request.model, "model_name", None)
            or getattr(request.model, "model", None)
            or type(request.model).__name__
        ),
        "request_message_count": len(request.messages),
        "request_delta": [
            message_to_dict(message) for message in _request_delta(request.messages)
        ],
    }
    if not any(getattr(message, "type", None) == "ai" for message in request.messages):
        base["system_message"] = (
            message_to_dict(request.system_message)
            if request.system_message is not None
            else None
        )
    try:
        response = await handler(request)
    except BaseException as error:
        await asyncio.to_thread(
            _append_record,
            {
                **base,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - started,
                "status": "error",
                "error": repr(error),
            },
        )
        raise
    await asyncio.to_thread(
        _append_record,
        {
            **base,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - started,
            "status": "success",
            "response": [message_to_dict(message) for message in response.result],
        },
    )
    return response
