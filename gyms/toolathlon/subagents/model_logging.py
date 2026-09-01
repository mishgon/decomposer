"""Crash-resistant append-only logging for subagent model calls."""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import message_to_dict


LOG_PATH_ENV = "TOOLATHLON_SUBAGENT_CALL_LOG"


def _request_delta(messages):
    last_ai = -1
    for index, message in enumerate(messages):
        if getattr(message, "type", None) == "ai":
            last_ai = index
    return messages if last_ai < 0 else messages[last_ai + 1 :]


def _append_record(record: dict) -> None:
    configured = os.environ.get(LOG_PATH_ENV)
    if not configured:
        return
    path = Path(configured)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
        fcntl.flock(output.fileno(), fcntl.LOCK_UN)


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
        # The initial call records the complete prompt. Later calls only need
        # messages after the preceding AI response (normally tool results), so
        # the JSONL stream remains reconstructable without quadratic growth.
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
        _append_record(
            {
                **base,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - started,
                "status": "error",
                "error": repr(error),
            }
        )
        raise
    _append_record(
        {
            **base,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - started,
            "status": "success",
            "response": [message_to_dict(message) for message in response.result],
        }
    )
    return response
