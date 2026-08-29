"""Immutable train/test partitioning for Gaia2 execution scenarios."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gyms.gaia2.dataset import sha256_file, validate_materialized_dataset
from gyms.gaia2.experiments import (
    DATASET_ID,
    DATASET_REVISION,
    DOMAIN,
    PARTITION_DATA_ROOT,
    SCENARIO_COUNT,
    SPLIT,
    SPLIT_MANIFEST_NAME,
    SPLIT_MANIFEST_SEED,
    TEST_SCENARIO_COUNT,
    TRAIN_SCENARIO_COUNT,
    Partition,
    dataset_revision_root,
)

SPLIT_MANIFEST_PATH = (
    Path(__file__).with_name("split_manifests") / f"{SPLIT_MANIFEST_NAME}.json"
)
SPLIT_MANIFEST_RELPATH = f"gyms/gaia2/split_manifests/{SPLIT_MANIFEST_NAME}.json"
SPLIT_MANIFEST_SHA256 = (
    "79f2725c48afc2c18db9a6266992bb27705ed86d5f8939dbee13964a768c5a20"
)
_SCENARIO_ID_RE = re.compile(r"scenario_universe_(\d+)_[a-z0-9]+")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _scenario_aggregate(scenarios: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for scenario in sorted(scenarios, key=lambda item: str(item["scenario_id"])):
        digest.update(str(scenario["scenario_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(scenario["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Gaia2 split manifest is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != SPLIT_MANIFEST_SHA256:
        raise ValueError(
            "Gaia2 split manifest checksum changed: "
            f"expected {SPLIT_MANIFEST_SHA256}, found {actual_sha256}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_dataset = {
        "id": DATASET_ID,
        "revision": DATASET_REVISION,
        "source_split": SPLIT,
        "domain": DOMAIN,
        "rows": SCENARIO_COUNT,
        "aggregate_sha256": (
            "600818deac34268a261cd846ef3019c6a4c87b068dccfea5e006cd72fe112518"
        ),
    }
    if manifest.get("schema_version") != 1:
        raise ValueError("Gaia2 split manifest has an unsupported schema version")
    if manifest.get("name") != SPLIT_MANIFEST_NAME:
        raise ValueError("Gaia2 split manifest has an unexpected name")
    if manifest.get("seed") != SPLIT_MANIFEST_SEED:
        raise ValueError("Gaia2 split manifest has an unexpected seed")
    if manifest.get("dataset") != expected_dataset:
        raise ValueError("Gaia2 split manifest dataset identity changed")
    if manifest.get("selection") != {
        "strategy": "complete-universe-holdout",
        "test_universes": [25, 26, 28],
    }:
        raise ValueError("Gaia2 split selection policy changed")

    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != SCENARIO_COUNT:
        raise ValueError("Gaia2 split manifest has an invalid scenario list")
    scenario_ids: set[str] = set()
    counts: Counter[str] = Counter()
    universes: dict[int, str] = {}
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id")
        match = _SCENARIO_ID_RE.fullmatch(str(scenario_id))
        if match is None or scenario_id in scenario_ids:
            raise ValueError(f"Invalid or duplicate Gaia2 scenario ID: {scenario_id!r}")
        universe = int(match.group(1))
        partition = scenario.get("partition")
        if scenario.get("universe") != universe or partition not in ("train", "test"):
            raise ValueError(f"Invalid Gaia2 split scenario assignment: {scenario!r}")
        previous = universes.setdefault(universe, partition)
        if previous != partition:
            raise ValueError(f"Universe {universe} is split across train and test")
        if not isinstance(scenario.get("size"), int) or scenario["size"] < 1:
            raise ValueError(f"Invalid Gaia2 split scenario size: {scenario!r}")
        digest = scenario.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid Gaia2 split scenario checksum: {scenario!r}")
        scenario_ids.add(scenario_id)
        counts[partition] += 1

    if counts != Counter(train=TRAIN_SCENARIO_COUNT, test=TEST_SCENARIO_COUNT):
        raise ValueError(f"Gaia2 split cardinalities changed: {dict(counts)}")
    assignments = {
        str(universe): partition for universe, partition in sorted(universes.items())
    }
    if manifest.get("universe_assignments") != assignments:
        raise ValueError("Gaia2 universe assignments changed")
    for partition in ("train", "test", "full"):
        selected = (
            scenarios
            if partition == "full"
            else [item for item in scenarios if item["partition"] == partition]
        )
        expected_partition = {
            "rows": len(selected),
            "universes": sorted({item["universe"] for item in selected}),
            "aggregate_sha256": _scenario_aggregate(selected),
        }
        if (manifest.get("partitions") or {}).get(partition) != expected_partition:
            raise ValueError(f"Gaia2 {partition} partition summary changed")
    return manifest


def partition_scenarios(
    manifest: Mapping[str, Any], partition: Partition
) -> list[dict[str, Any]]:
    scenarios = list(manifest["scenarios"])
    if partition == "full":
        return scenarios
    if partition not in ("train", "test"):
        raise ValueError(f"Unknown Gaia2 partition: {partition!r}")
    return [item for item in scenarios if item["partition"] == partition]


def partition_scenario_ids(partition: Partition) -> tuple[str, ...]:
    manifest = load_split_manifest()
    return tuple(
        item["scenario_id"] for item in partition_scenarios(manifest, partition)
    )


def validate_split_against_source(
    split_manifest: Mapping[str, Any], source_manifest: Mapping[str, Any]
) -> None:
    if split_manifest["dataset"]["aggregate_sha256"] != source_manifest.get(
        "aggregate_sha256"
    ):
        raise ValueError("Gaia2 source dataset checksum does not match split manifest")
    source = {
        item["scenario_id"]: {
            "size": item["size"],
            "sha256": item["sha256"],
        }
        for item in source_manifest["scenarios"]
    }
    expected = {
        item["scenario_id"]: {
            "size": item["size"],
            "sha256": item["sha256"],
        }
        for item in split_manifest["scenarios"]
    }
    if source != expected:
        raise ValueError("Gaia2 source scenarios do not match split manifest")


def _views_manifest_path(root: Path) -> Path:
    return root / "partition_views_manifest.json"


def validate_partition_view(
    partition: Partition,
    *,
    root: Path = PARTITION_DATA_ROOT,
    source_root: Path | None = None,
) -> dict[str, Any]:
    source_revision_root = source_root or dataset_revision_root()
    source_manifest = validate_materialized_dataset(source_revision_root)
    split_manifest = load_split_manifest()
    validate_split_against_source(split_manifest, source_manifest)
    selected = partition_scenarios(split_manifest, partition)
    if partition == "full":
        return {
            "partition": partition,
            "rows": len(selected),
            "aggregate_sha256": _scenario_aggregate(selected),
            "scenario_directory": str(source_revision_root / SPLIT / DOMAIN),
        }

    views_path = _views_manifest_path(root)
    if not views_path.is_file():
        raise FileNotFoundError(
            f"Prepared Gaia2 partition manifest is missing: {views_path}"
        )
    views = json.loads(views_path.read_text(encoding="utf-8"))
    expected_view = {
        "rows": len(selected),
        "aggregate_sha256": _scenario_aggregate(selected),
        "scenario_directory": str(root / partition / DOMAIN),
    }
    if views.get("schema_version") != 1:
        raise ValueError("Prepared Gaia2 partition views use an unsupported schema")
    if views.get("split_manifest") != {
        "name": SPLIT_MANIFEST_NAME,
        "path": SPLIT_MANIFEST_RELPATH,
        "sha256": SPLIT_MANIFEST_SHA256,
    }:
        raise ValueError("Prepared Gaia2 partition views use an unexpected split")
    if (views.get("partitions") or {}).get(partition) != expected_view:
        raise ValueError(f"Prepared Gaia2 {partition} view manifest changed")

    source_directory = source_revision_root / SPLIT / DOMAIN
    scenario_directory = root / partition / DOMAIN
    expected_files = {f"{item['scenario_id']}.json" for item in selected}
    actual_files = {path.name for path in scenario_directory.glob("*.json")}
    if actual_files != expected_files:
        raise ValueError(f"Prepared Gaia2 {partition} view contains unexpected files")
    for item in selected:
        filename = f"{item['scenario_id']}.json"
        path = scenario_directory / filename
        source_path = source_directory / filename
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Prepared Gaia2 partition scenario changed: {path}")
        if path.samefile(source_path):
            raise ValueError(
                f"Prepared Gaia2 partition scenario is not isolated: {path}"
            )
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Prepared Gaia2 partition checksum changed: {path}")
    return {"partition": partition, **expected_view}


def materialize_partition_views(
    *,
    root: Path = PARTITION_DATA_ROOT,
    source_root: Path | None = None,
) -> dict[str, Any]:
    source_revision_root = source_root or dataset_revision_root()
    source_manifest = validate_materialized_dataset(source_revision_root)
    split_manifest = load_split_manifest()
    validate_split_against_source(split_manifest, source_manifest)
    if root.exists():
        return {
            partition: validate_partition_view(
                partition, root=root, source_root=source_revision_root
            )
            for partition in ("train", "test")
        }

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.with_name(f".{root.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(
            f"Incomplete Gaia2 partition preparation exists: {temporary}"
        )
    source_directory = source_revision_root / SPLIT / DOMAIN
    try:
        partition_values: dict[str, Any] = {}
        for partition in ("train", "test"):
            selected = partition_scenarios(split_manifest, partition)
            destination = temporary / partition / DOMAIN
            destination.mkdir(parents=True)
            for item in selected:
                filename = f"{item['scenario_id']}.json"
                shutil.copy2(source_directory / filename, destination / filename)
            partition_values[partition] = {
                "rows": len(selected),
                "aggregate_sha256": _scenario_aggregate(selected),
                "scenario_directory": str(root / partition / DOMAIN),
            }
        _atomic_json(
            _views_manifest_path(temporary),
            {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "source_dataset_manifest": str(
                    source_revision_root / "dataset_manifest.json"
                ),
                "split_manifest": {
                    "name": SPLIT_MANIFEST_NAME,
                    "path": SPLIT_MANIFEST_RELPATH,
                    "sha256": SPLIT_MANIFEST_SHA256,
                },
                "partitions": partition_values,
            },
        )
        os.rename(temporary, root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        partition: validate_partition_view(
            partition, root=root, source_root=source_revision_root
        )
        for partition in ("train", "test")
    }
