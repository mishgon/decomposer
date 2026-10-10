"""Shared adapter result types and helpers for canonical SFT preparation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..schema import (
    EXCLUSION_REASONS,
    CanonicalRollout,
    CanonicalSource,
    JsonObject,
    SourceSpec,
)


@dataclass(frozen=True)
class AdapterReadResult:
    records: tuple[CanonicalRollout, ...]
    source_manifest: JsonObject
    counts: Counter[str]


def empty_counts() -> Counter[str]:
    return Counter({reason: 0 for reason in EXCLUSION_REASONS})


def load_json(path: Path) -> JsonObject:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected {path} to contain a JSON object.")
    return value


def check_native_rollouts(source: SourceSpec, native_rollouts: int) -> None:
    if native_rollouts != source.expected_native_rollouts:
        raise ValueError(
            f"Source {source.id!r} expected {source.expected_native_rollouts} "
            f"native rollouts, found {native_rollouts}."
        )


def canonical_source(
    source: SourceSpec, adapter_version: int, *, task_id: str, rollout_id: str
) -> CanonicalSource:
    return CanonicalSource(
        adapter=source.adapter,
        adapter_version=adapter_version,
        source_id=source.id,
        benchmark=source.benchmark,
        environment=source.environment,
        partition=source.partition,
        teacher=source.teacher,
        task_id=task_id,
        rollout_id=rollout_id,
    )
