"""Native rollout adapters for canonical SFT preparation."""

from __future__ import annotations

from typing import Protocol

from ..schema import SelectionSpec, SourceSpec
from .nemo_gym import ADAPTER_VERSION as NEMO_GYM_ADAPTER_VERSION
from .nemo_gym import AdapterReadResult, read_nemo_gym_source


class AdapterReader(Protocol):
    def __call__(
        self,
        source: SourceSpec,
        selection: SelectionSpec,
        *,
        system_prompt: str,
    ) -> AdapterReadResult: ...


ADAPTERS: dict[str, AdapterReader] = {"nemo_gym": read_nemo_gym_source}
ADAPTER_VERSIONS = {"nemo_gym": NEMO_GYM_ADAPTER_VERSION}

__all__ = [
    "ADAPTERS",
    "ADAPTER_VERSIONS",
    "AdapterReader",
    "AdapterReadResult",
    "NEMO_GYM_ADAPTER_VERSION",
    "read_nemo_gym_source",
]
