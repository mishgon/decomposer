"""Pinned Gaia2 dataset materialization and integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .experiments import (
    DATASET_ID,
    DATASET_REVISION,
    DOMAIN,
    FILESYSTEM_DATASET_ID,
    FILESYSTEM_DATASET_REVISION,
    SCENARIO_COUNT,
    SPLIT,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_row(row: Mapping[str, Any]) -> tuple[str, str]:
    scenario_id = row.get("scenario_id")
    data = row.get("data")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("Gaia2 row has no scenario_id")
    if Path(scenario_id).name != scenario_id:
        raise ValueError(f"Unsafe Gaia2 scenario_id: {scenario_id!r}")
    if row.get("split") != SPLIT:
        raise ValueError(
            f"Gaia2 row {scenario_id} belongs to split {row.get('split')!r}"
        )
    if not isinstance(data, str):
        raise TypeError(f"Gaia2 row {scenario_id} data is not serialized JSON")
    parsed = json.loads(data)
    embedded_id = ((parsed.get("metadata") or {}).get("definition") or {}).get(
        "scenario_id"
    )
    if embedded_id != scenario_id:
        raise ValueError(
            f"Gaia2 row ID mismatch: column={scenario_id!r}, embedded={embedded_id!r}"
        )
    return scenario_id, data


def write_materialized_dataset(
    rows: Iterable[Mapping[str, Any]],
    destination_revision_root: Path,
    *,
    gaia2_revision: str,
) -> dict[str, Any]:
    """Write one immutable dataset revision atomically."""

    if destination_revision_root.exists():
        return validate_materialized_dataset(destination_revision_root)

    destination_revision_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_revision_root.with_name(
        f".{destination_revision_root.name}.tmp.{os.getpid()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    scenario_directory = temporary / SPLIT / DOMAIN
    scenario_directory.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        for row in rows:
            scenario_id, data = _validate_row(row)
            if scenario_id in seen:
                raise ValueError(f"Duplicate Gaia2 scenario_id: {scenario_id}")
            seen.add(scenario_id)
            encoded = data.encode("utf-8")
            target = scenario_directory / f"{scenario_id}.json"
            target.write_bytes(encoded)
            digest = sha256_bytes(encoded)
            entries.append(
                {
                    "scenario_id": scenario_id,
                    "filename": target.name,
                    "size": len(encoded),
                    "sha256": digest,
                }
            )
        entries.sort(key=lambda item: item["scenario_id"])
        if len(entries) != SCENARIO_COUNT:
            raise ValueError(
                f"Expected {SCENARIO_COUNT} Gaia2 scenarios, found {len(entries)}"
            )
        aggregate = hashlib.sha256()
        for entry in entries:
            encoded = (scenario_directory / entry["filename"]).read_bytes()
            aggregate.update(entry["scenario_id"].encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(encoded)
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "domain": DOMAIN,
            "split": SPLIT,
            "rows": len(entries),
            "aggregate_sha256": aggregate.hexdigest(),
            "gaia2_revision": gaia2_revision,
            "scenario_directory": str(destination_revision_root / SPLIT / DOMAIN),
            "scenarios": entries,
        }
        (temporary / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            os.rename(temporary, destination_revision_root)
        except FileExistsError:
            shutil.rmtree(temporary)
        return validate_materialized_dataset(destination_revision_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_materialized_dataset(
    revision_root: Path,
) -> dict[str, Any]:
    manifest_path = revision_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared Gaia2 manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "domain": DOMAIN,
        "split": SPLIT,
        "rows": SCENARIO_COUNT,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Prepared Gaia2 manifest mismatch: {mismatches}")
    scenario_directory = revision_root / SPLIT / DOMAIN
    if Path(manifest.get("scenario_directory", "")) != scenario_directory:
        raise ValueError("Prepared Gaia2 manifest points at an unexpected directory")
    entries = manifest.get("scenarios")
    if not isinstance(entries, list) or len(entries) != SCENARIO_COUNT:
        raise ValueError("Prepared Gaia2 manifest has an invalid scenario list")
    expected_files: set[str] = set()
    aggregate = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["scenario_id"]):
        filename = entry.get("filename")
        scenario_id = entry.get("scenario_id")
        if filename != f"{scenario_id}.json":
            raise ValueError(f"Invalid Gaia2 manifest entry: {entry}")
        path = scenario_directory / filename
        expected_files.add(filename)
        if not path.is_file() or path.stat().st_size != entry.get("size"):
            raise ValueError(f"Prepared Gaia2 scenario changed or is missing: {path}")
        encoded = path.read_bytes()
        if sha256_bytes(encoded) != entry.get("sha256"):
            raise ValueError(f"Prepared Gaia2 scenario checksum changed: {path}")
        aggregate.update(str(scenario_id).encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(encoded)
    actual_files = {path.name for path in scenario_directory.glob("*.json")}
    if actual_files != expected_files:
        raise ValueError("Prepared Gaia2 scenario directory contains unexpected files")
    if aggregate.hexdigest() != manifest.get("aggregate_sha256"):
        raise ValueError("Prepared Gaia2 aggregate checksum changed")
    return manifest


def load_huggingface_rows(*, cache_dir: Path) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return load_dataset(
        DATASET_ID,
        DOMAIN,
        split=SPLIT,
        revision=DATASET_REVISION,
        cache_dir=str(cache_dir),
    )


def write_filesystem_manifest(revision_root: Path) -> dict[str, Any]:
    """Record every locally mirrored Gaia2 filesystem asset."""

    directory = revision_root / "demo_filesystem"
    if not directory.is_dir():
        raise FileNotFoundError(f"Gaia2 filesystem directory is missing: {directory}")
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        entries.append({"path": relative, "size": size, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
    if not entries:
        raise ValueError(f"Gaia2 filesystem contains no files: {directory}")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": FILESYSTEM_DATASET_ID,
        "dataset_revision": FILESYSTEM_DATASET_REVISION,
        "directory": "demo_filesystem",
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "aggregate_sha256": aggregate.hexdigest(),
    }
    path = revision_root / "filesystem_manifest.json"
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return validate_materialized_filesystem(revision_root)


def validate_materialized_filesystem(revision_root: Path) -> dict[str, Any]:
    manifest_path = revision_root / "filesystem_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared Gaia2 filesystem manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": FILESYSTEM_DATASET_ID,
        "dataset_revision": FILESYSTEM_DATASET_REVISION,
        "directory": "demo_filesystem",
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Prepared Gaia2 filesystem manifest mismatch: {mismatches}")
    directory = revision_root / "demo_filesystem"
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Prepared Gaia2 filesystem manifest has no files")
    expected_files: set[str] = set()
    aggregate = hashlib.sha256()
    total_bytes = 0
    for entry in sorted(entries, key=lambda item: item["path"]):
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Invalid Gaia2 filesystem manifest entry: {entry}")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Unsafe Gaia2 filesystem path: {relative!r}")
        path = directory / candidate
        expected_files.add(candidate.as_posix())
        size = entry.get("size")
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Prepared Gaia2 filesystem asset changed: {path}")
        digest = sha256_file(path)
        if digest != entry.get("sha256"):
            raise ValueError(f"Prepared Gaia2 filesystem checksum changed: {path}")
        aggregate.update(candidate.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        total_bytes += size
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Prepared Gaia2 filesystem contains unexpected files")
    if manifest.get("file_count") != len(entries):
        raise ValueError("Prepared Gaia2 filesystem file count changed")
    if manifest.get("total_bytes") != total_bytes:
        raise ValueError("Prepared Gaia2 filesystem byte count changed")
    if manifest.get("aggregate_sha256") != aggregate.hexdigest():
        raise ValueError("Prepared Gaia2 filesystem aggregate checksum changed")
    return manifest
