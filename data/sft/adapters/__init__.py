"""Native rollout adapters for canonical SFT preparation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..schema import JsonObject, SelectionSpec, SourceSpec
from .base import AdapterReadResult
from .gaia2 import ADAPTER_VERSION as GAIA2_ADAPTER_VERSION
from .gaia2 import read_gaia2_source
from .nemo_gym import ADAPTER_VERSION as NEMO_GYM_ADAPTER_VERSION
from .nemo_gym import read_nemo_gym_source
from .toolathlon_gym import ADAPTER_VERSION as TOOLATHLON_GYM_ADAPTER_VERSION
from .toolathlon_gym import read_toolathlon_gym_source


class AdapterReader(Protocol):
    def __call__(
        self,
        source: SourceSpec,
        selection: SelectionSpec,
        *,
        system_prompt: str,
        canonical_tools: Sequence[JsonObject] | None = None,
        canonical_subagent_type_ids: frozenset[str] = frozenset(),
    ) -> AdapterReadResult: ...


ADAPTERS: dict[str, AdapterReader] = {
    "gaia2": read_gaia2_source,
    "nemo_gym": read_nemo_gym_source,
    "toolathlon_gym": read_toolathlon_gym_source,
}
ADAPTER_VERSIONS = {
    "gaia2": GAIA2_ADAPTER_VERSION,
    "nemo_gym": NEMO_GYM_ADAPTER_VERSION,
    "toolathlon_gym": TOOLATHLON_GYM_ADAPTER_VERSION,
}

__all__ = [
    "ADAPTERS",
    "ADAPTER_VERSIONS",
    "AdapterReader",
    "AdapterReadResult",
    "GAIA2_ADAPTER_VERSION",
    "NEMO_GYM_ADAPTER_VERSION",
    "TOOLATHLON_GYM_ADAPTER_VERSION",
    "read_gaia2_source",
    "read_nemo_gym_source",
    "read_toolathlon_gym_source",
]
