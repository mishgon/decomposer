"""JSONL trace reader with type-based partitioning."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..models.trace import (
    AuditSnapshot,
    DecomposerSummary,
    DelegationEnd,
    DelegationStart,
    GradingResult,
    MediaLoad,
    ToolDispatch,
    TraceEnd,
    TraceMessage,
    TraceStart,
)

_EVENT_MAP = {
    "trace_start": TraceStart,
    "message": TraceMessage,
    "tool_dispatch": ToolDispatch,
    "audit_snapshot": AuditSnapshot,
    "media_load": MediaLoad,
    "trace_end": TraceEnd,
    "grading_result": GradingResult,
    "delegation_start": DelegationStart,
    "delegation_end": DelegationEnd,
    "decomposer_summary": DecomposerSummary,
    "compact": None,  # skipped during load_trace
}


def read_events(path: str | Path) -> Iterator:
    """Parse each JSONL line by its ``type`` discriminator field."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            event_type = raw.get("type")
            cls = _EVENT_MAP.get(event_type)
            if cls is None:
                if event_type == "compact":
                    continue
                raise ValueError(f"Unknown trace event type: {event_type!r}")
            yield cls.model_validate(raw)


def load_trace(
    path: str | Path,
) -> tuple[TraceStart, list[TraceMessage], list[ToolDispatch], list[MediaLoad], TraceEnd | None, dict[str, dict]]:
    """Load a full trace file and partition by event type.

    Returns (start, messages, dispatches, media_events, end, audit_data).
    audit_data is keyed by service_name from AuditSnapshot events.
    GradingResult and decomposer analysis events are silently skipped.
    """
    start: TraceStart | None = None
    messages: list[TraceMessage] = []
    dispatches: list[ToolDispatch] = []
    media_events: list[MediaLoad] = []
    end: TraceEnd | None = None
    audit_data: dict[str, dict] = {}

    for event in read_events(path):
        match event:
            case TraceStart():
                start = event
            case TraceMessage():
                messages.append(event)
            case ToolDispatch():
                dispatches.append(event)
            case AuditSnapshot():
                audit_data[event.service_name] = event.audit_data
            case MediaLoad():
                media_events.append(event)
            case TraceEnd():
                end = event
            case GradingResult() | DelegationStart() | DelegationEnd() | DecomposerSummary():
                pass

    if start is None:
        raise ValueError(f"No TraceStart event found in {path}")
    return start, messages, dispatches, media_events, end, audit_data


def load_trace_for_grading(
    path: str | Path,
    *,
    include_decomposer_sidecars: bool = True,
) -> tuple[TraceStart, list[TraceMessage], list[ToolDispatch], list[MediaLoad], TraceEnd | None, dict[str, dict]]:
    """Load a trace for grading.

    Decomposer traces keep Executor environment calls in sidecar traces. Task
    graders use ``ToolDispatch`` records for tool-use gates and robustness, so
    grading must count those sidecar dispatches while leaving the manager
    transcript unchanged for judge prompts.
    """
    trace_path = Path(path)
    start, messages, dispatches, media_events, end, audit_data = load_trace(trace_path)
    if not include_decomposer_sidecars or start.run_mode != "decomposer":
        return start, messages, dispatches, media_events, end, audit_data

    sidecar_dispatches: list[ToolDispatch] = []
    for event in read_events(trace_path):
        if not isinstance(event, DelegationEnd) or not event.sidecar_trace:
            continue
        sidecar_path = Path(event.sidecar_trace)
        if not sidecar_path.is_absolute():
            sidecar_path = trace_path.parent / sidecar_path
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"Missing decomposer executor sidecar trace for grading: {sidecar_path}"
            )
        for sidecar_event in read_events(sidecar_path):
            if isinstance(sidecar_event, ToolDispatch):
                sidecar_dispatches.append(sidecar_event)

    if sidecar_dispatches:
        dispatches = [*dispatches, *sidecar_dispatches]
    return start, messages, dispatches, media_events, end, audit_data
