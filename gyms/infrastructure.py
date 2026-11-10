"""Identify service failures without treating agent limits as infrastructure errors."""
import json
import re


def classify(error):
    text = str(error)
    if any(name in text for name in ("AuthenticationError", "PermissionDeniedError", "insufficient_credits")) or re.search(r"\b(401|402)\b", text):
        return "credentials_or_quota"
    if any(name in text for name in ("APIConnectionError", "OpenAIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout", "RemoteProtocolError", "RateLimitError", "InternalServerError")) or re.search(r"\b(429|500|502|503|504)\b", text):
        return "service_unavailable"
    return None


def failures(directory):
    errors = []
    for filename in ("model_calls.jsonl", "web_calls.jsonl"):
        path = directory / filename
        if not path.exists():
            continue
        for line in path.read_text().split("\n"):
            if '"error"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:  # A concurrent writer may not have finished its last line.
                continue
            if row.get("error") and (kind := classify(row["error"])):
                errors.append({"kind": kind, "source": filename, "error": row["error"]})
    path = directory / "trace.json"
    if path.exists():
        trace = json.loads(path.read_text())
        for error in [trace.get("error"), *[r.get("error") for r in trace.get("agent_runs", {}).values()]]:
            if error and (kind := classify(error)):
                errors.append({"kind": kind, "source": "trace.json", "error": error})
    return errors
