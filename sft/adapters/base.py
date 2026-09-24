"""Shared adapter result types for canonical SFT preparation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..schema import CanonicalRollout, JsonObject


@dataclass(frozen=True)
class AdapterReadResult:
    records: tuple[CanonicalRollout, ...]
    source_manifest: JsonObject
    counts: Counter[str]
