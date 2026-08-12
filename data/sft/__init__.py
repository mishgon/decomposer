"""Versioned, benchmark-neutral Decomposer SFT dataset preparation."""

from .builder import PreparedDataset, load_build_spec, prepare_dataset
from .schema import BuildSpec, CanonicalRollout

__all__ = [
    "BuildSpec",
    "CanonicalRollout",
    "PreparedDataset",
    "load_build_spec",
    "prepare_dataset",
]
