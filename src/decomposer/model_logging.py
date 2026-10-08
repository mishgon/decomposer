"""Durable JSONL records and incremental model request serialization."""
import fcntl
import json
import os
from pathlib import Path


def request_delta(messages):
    last_ai = -1
    for index, message in enumerate(messages):
        if message.type == "ai":
            last_ai = index
    return messages[last_ai + 1:]


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
        fcntl.flock(output.fileno(), fcntl.LOCK_UN)
