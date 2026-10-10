"""Migrate completed Workplace outputs to explicit call-limit identities.

The command is a dry run unless ``--apply`` is passed. It never overwrites a
destination. Rollouts, metrics, logs, and attempt directories are moved without
modification; only the two top-level identity JSON files are rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gyms.workplace_assistant.experiments import RESULTS_ROOT


MIGRATION_ID = "workplace-call-limit-identities-v1"
MIGRATION_MANIFEST = f".{MIGRATION_ID}.json"


@dataclass(frozen=True)
class Migration:
    source: str
    destination: str
    source_experiment: str
    destination_experiment: str
    num_repeats: int
    max_steps: int


MIGRATIONS = (
    Migration(
        "deepseek-v4-flash-0731-n3",
        "deepseek-v4-flash-0731-calls6-n3",
        "deepseek-v4-flash-0731",
        "deepseek-v4-flash-0731",
        3,
        6,
    ),
    Migration(
        "gemma4-e4b-it-thinking-n3",
        "gemma4-e4b-it-thinking-calls6-n3",
        "gemma4-e4b-it-thinking",
        "gemma4-e4b-it-thinking",
        3,
        6,
    ),
    Migration(
        "grpo-4b-v26-lr15-n5",
        "grpo-4b-v26-lr15-calls6-n5",
        "grpo-4b-v26-lr15",
        "grpo-4b-v26-lr15",
        5,
        6,
    ),
    Migration(
        "grpo-4b-v30-lr15-n5",
        "grpo-4b-v30-lr15-calls6-n5",
        "grpo-4b-v30-lr15",
        "grpo-4b-v30-lr15",
        5,
        6,
    ),
    Migration(
        "qwen35-2b-base-non-thinking-n3",
        "qwen35-2b-base-non-thinking-calls6-n3",
        "qwen35-2b-base-non-thinking",
        "qwen35-2b-base-non-thinking",
        3,
        6,
    ),
    Migration(
        "qwen35-4b-base-non-thinking-n3",
        "qwen35-4b-base-non-thinking-calls6-n3",
        "qwen35-4b-base-non-thinking",
        "qwen35-4b-base-non-thinking",
        3,
        6,
    ),
    Migration(
        "qwen35-9b-base-non-thinking-n3",
        "qwen35-9b-base-non-thinking-calls6-n3",
        "qwen35-9b-base-non-thinking",
        "qwen35-9b-base-non-thinking",
        3,
        6,
    ),
    Migration(
        "qwen35-4b-base-non-thinking-maxsteps100-n3",
        "qwen35-4b-base-non-thinking-calls100-n3",
        "qwen35-4b-base-non-thinking-maxsteps100",
        "qwen35-4b-base-non-thinking",
        3,
        100,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _replace_path_strings(value: Any, source: str, destination: str) -> Any:
    if isinstance(value, str):
        return value.replace(source, destination)
    if isinstance(value, list):
        return [_replace_path_strings(item, source, destination) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_path_strings(item, source, destination)
            for key, item in value.items()
        }
    return value


def _load_and_validate_metadata(
    directory: Path, spec: Migration
) -> dict[str, dict[str, Any]]:
    result = {}
    for filename in ("run_status.json", ".eval_done.json"):
        path = directory / filename
        if not path.is_file():
            raise RuntimeError(f"Missing completed-run metadata: {path}")
        value = json.loads(path.read_text())
        expected = {
            "state": "complete",
            "kind": "simple",
            "purpose": "evaluation",
            "split": "validation",
            "num_repeats": spec.num_repeats,
            "limit": None,
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise RuntimeError(
                    f"Unexpected {key} in {path}: {value.get(key)!r}; "
                    f"expected {expected_value!r}"
                )
        if value.get("experiment") not in {
            spec.source_experiment,
            spec.destination_experiment,
        }:
            raise RuntimeError(
                f"Unexpected experiment in {path}: {value.get('experiment')!r}"
            )
        result[filename] = value
    return result


def _raw_file_manifest(directory: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in {"run_status.json", ".eval_done.json"}:
            continue
        relative = str(path.relative_to(directory))
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return files


def _rewrite_metadata(directory: Path, spec: Migration) -> None:
    source_path = str(directory.parent / spec.source)
    destination_path = str(directory.parent / spec.destination)
    metadata = _load_and_validate_metadata(directory, spec)
    for filename, original in metadata.items():
        updated = _replace_path_strings(original, source_path, destination_path)
        updated.update(
            {
                "schema_version": max(int(updated.get("schema_version", 0)), 4),
                "experiment": spec.destination_experiment,
                "run_name": spec.destination,
                "simple_agent_max_steps": spec.max_steps,
                "artifact_identity_migration": {
                    "id": MIGRATION_ID,
                    "source_directory_name": spec.source,
                    "source_experiment": spec.source_experiment,
                },
            }
        )
        if spec.source_experiment != spec.destination_experiment:
            updated["preparation_manifest"] = str(
                directory.parent.parent.parent
                / "data"
                / "workplace_assistant"
                / "manifests"
                / "validation"
                / f"{spec.destination_experiment}.json"
            )
        _atomic_json(directory / filename, updated)


def migrate(root: Path, *, apply: bool) -> dict[str, Any]:
    manifest_path = root / MIGRATION_MANIFEST
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("migration_id") != MIGRATION_ID:
            raise RuntimeError(f"Unexpected migration manifest: {manifest_path}")
        if apply:
            existing_records = {
                record["destination"]: record for record in existing.get("records", [])
            }
            for spec in MIGRATIONS:
                source = root / spec.source
                destination = root / spec.destination
                if source.exists() or not destination.is_dir():
                    raise RuntimeError(
                        f"Applied migration state is inconsistent for {spec.source}"
                    )
                _load_and_validate_metadata(destination, spec)
                record = existing_records.get(str(destination))
                if record is None or record.get("raw_files") != _raw_file_manifest(
                    destination
                ):
                    raise RuntimeError(
                        f"Applied migration integrity check failed for {destination}"
                    )
            return existing

    records = []
    for spec in MIGRATIONS:
        source = root / spec.source
        destination = root / spec.destination
        if source.exists() and destination.exists():
            raise RuntimeError(
                f"Both source and destination exist: {source}, {destination}"
            )
        if not source.exists() and not destination.exists():
            raise FileNotFoundError(
                f"Neither source nor destination exists for {spec.source}"
            )

        current = source if source.exists() else destination
        _load_and_validate_metadata(current, spec)
        raw_files = _raw_file_manifest(current)
        action = "already-migrated" if current == destination else "move"
        if apply and current == source:
            os.replace(source, destination)
            current = destination
        if apply:
            _rewrite_metadata(current, spec)
            if _raw_file_manifest(current) != raw_files:
                raise RuntimeError(
                    f"Raw artifact integrity changed while migrating {spec.source}"
                )
        records.append(
            {
                "source": str(source),
                "destination": str(destination),
                "action": action,
                "source_experiment": spec.source_experiment,
                "destination_experiment": spec.destination_experiment,
                "num_repeats": spec.num_repeats,
                "simple_agent_max_steps": spec.max_steps,
                "raw_files": raw_files,
            }
        )

    manifest = {
        "schema_version": 1,
        "migration_id": MIGRATION_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "applied": apply,
        "records": records,
    }
    if apply:
        _atomic_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=RESULTS_ROOT / "validation",
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(migrate(args.root.resolve(), apply=args.apply), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
