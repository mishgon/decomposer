"""Versioned tau2 task pools for the Decomposer.

A pool is ``<name>.json`` (``{domain: [task_id, ...]}``) next to ``<name>.meta.json``,
built by ``build_pool.py`` from a recipe. Consumers refer to pools by name and check
the recorded sha256, so a pool file edited after it was built is rejected instead of
silently changing what an experiment ran on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

POOLS_DIR = Path(__file__).resolve().parent

TaskKey = tuple[str, str]


def pool_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def resolve_pool(name_or_path: str | Path) -> Path:
    """A pool name (``decomposer_train_v2``) or a path to its ``.json``."""
    candidate = Path(name_or_path)
    if candidate.suffix == ".json":
        return candidate.resolve()
    return POOLS_DIR / f"{name_or_path}.json"


def load_pool(name_or_path: str | Path) -> tuple[list[TaskKey], dict[str, Any]]:
    """Pool tasks in file order (domains sorted, ids sorted) and the pool's meta."""
    path = resolve_pool(name_or_path)
    meta_path = path.with_name(path.stem + ".meta.json")
    if not path.is_file():
        raise FileNotFoundError(f"Unknown task pool: {path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Task pool {path.name} has no meta file at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual = pool_sha256(path)
    if meta.get("sha256") != actual:
        raise ValueError(
            f"Task pool {path.name} does not match its meta (sha256 {actual} != "
            f"{meta.get('sha256')}). Rebuild it with build_pool.py instead of editing it."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = [(domain, str(task_id)) for domain, ids in raw.items() for task_id in ids]
    return tasks, meta
