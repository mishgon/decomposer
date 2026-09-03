"""Token-usage accounting for Toolathlon Gym Decomposer traces."""

from __future__ import annotations

from typing import Any, Iterable


TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
)


def _message_data(message: dict[str, Any]) -> dict[str, Any]:
    data = message.get("data")
    return data if isinstance(data, dict) else message


def _usage_for_message(message: dict[str, Any]) -> dict[str, Any] | None:
    data = _message_data(message)
    usage = data.get("usage_metadata")
    if not isinstance(usage, dict):
        return None
    input_details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}
    response_metadata = data.get("response_metadata") or {}
    token_usage = response_metadata.get("token_usage") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cache_read_tokens": int(input_details.get("cache_read") or 0),
        "cache_creation_tokens": int(input_details.get("cache_creation") or 0),
        "reasoning_tokens": int(output_details.get("reasoning") or 0),
        "cost": token_usage.get("cost"),
    }


def summarize_messages(messages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {field: 0 for field in TOKEN_FIELDS}
    totals.update(model_responses=0, responses_with_usage=0, cost=0.0)
    cost_reported = False
    for message in messages:
        data = _message_data(message)
        if message.get("type") != "ai" and data.get("type") != "ai":
            continue
        totals["model_responses"] += 1
        usage = _usage_for_message(message)
        if usage is None:
            continue
        totals["responses_with_usage"] += 1
        for field in TOKEN_FIELDS:
            totals[field] += usage[field]
        if isinstance(usage["cost"], (int, float)):
            totals["cost"] += float(usage["cost"])
            cost_reported = True
    if not cost_reported:
        totals["cost"] = None
    return totals


def build_usage_summary(
    decomposer_messages: list[dict[str, Any]],
    subagent_runs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    decomposer = summarize_messages(decomposer_messages)
    subagents = {
        run_id: {
            "subagent_type_id": run.get("subagent_type_id"),
            "status": run.get("status"),
            **summarize_messages(run.get("messages") or []),
        }
        for run_id, run in subagent_runs.items()
    }
    totals: dict[str, Any] = {field: decomposer[field] for field in TOKEN_FIELDS}
    totals.update(
        model_responses=decomposer["model_responses"],
        responses_with_usage=decomposer["responses_with_usage"],
        cost=decomposer["cost"] or 0.0,
    )
    cost_reported = decomposer["cost"] is not None
    for usage in subagents.values():
        for field in (*TOKEN_FIELDS, "model_responses", "responses_with_usage"):
            totals[field] += usage[field]
        if usage["cost"] is not None:
            totals["cost"] += usage["cost"]
            cost_reported = True
    if not cost_reported:
        totals["cost"] = None
    totals["subagent_runs"] = len(subagents)
    return {"decomposer": decomposer, "subagents": subagents, "totals": totals}
