"""Append model request deltas and responses before an episode finishes."""
import asyncio
import os
import time
from uuid import uuid4

from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import message_to_dict
from decomposer.model_logging import append_record, request_delta


@wrap_model_call
async def log_model_call(request, handler):
    path = os.environ["BROWSECOMP_MODEL_LOG"]
    started = time.time()
    call_id = uuid4().hex
    await asyncio.to_thread(append_record, path, {
        "call_id": call_id, "status": "started", "started_at": started,
        "model": request.model.model_name,
        "request_delta": [message_to_dict(m) for m in request_delta(request.messages)],
        "system_message": message_to_dict(request.system_message) if request.system_message else None,
    })
    try:
        response = await handler(request)
    except BaseException as error:
        await asyncio.to_thread(append_record, path, {"call_id": call_id, "status": "error", "error": repr(error),
                             "duration_seconds": time.time() - started})
        raise
    await asyncio.to_thread(append_record, path, {"call_id": call_id, "status": "success",
                         "duration_seconds": time.time() - started,
                         "response": [message_to_dict(m) for m in response.result]})
    return response
