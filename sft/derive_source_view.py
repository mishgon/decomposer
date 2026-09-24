"""Derive an immutable, checksum-pinned single-source candidate dataset view."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .builder import compute_dataset_fingerprint
from .schema import (
    CANONICAL_SCHEMA_VERSION,
    MANIFEST_FORMAT_VERSION,
    CanonicalRollout,
    canonical_json,
    sha256_file,
    sha256_text,
)

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class DerivedSourceView:
    output_dir: Path
    train_path: Path
    validation_path: Path
    manifest_path: Path
    manifest: JsonObject


def _read_manifest(path: Path) -> JsonObject:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Dataset manifest {path} must be an object.")
    if value.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"Dataset manifest format_version must be {MANIFEST_FORMAT_VERSION}."
        )
    if value.get("canonical_schema_version") != CANONICAL_SCHEMA_VERSION:
        raise ValueError("Dataset manifest has an unsupported canonical schema.")
    identity = value.get("dataset")
    if not isinstance(identity, Mapping):
        raise ValueError("Dataset manifest has no dataset identity.")
    fingerprint = identity.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Dataset manifest has no valid fingerprint.")
    actual = compute_dataset_fingerprint(value)
    if fingerprint != actual:
        raise ValueError(
            f"Parent dataset fingerprint is {fingerprint}, but resolves to {actual}."
        )
    return value


def _validate_prepared_file(
    manifest: Mapping[str, Any], path: Path, *, filename: str
) -> None:
    prepared = manifest.get("prepared_files")
    metadata = prepared.get(filename) if isinstance(prepared, Mapping) else None
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Parent manifest has no metadata for {filename}.")
    if not path.is_file():
        raise FileNotFoundError(f"Parent prepared file does not exist: {path}")
    if metadata.get("bytes") != path.stat().st_size:
        raise ValueError(f"Parent prepared file byte size changed: {path}")
    if metadata.get("sha256") != sha256_file(path):
        raise ValueError(f"Parent prepared file checksum changed: {path}")


def _read_selected_lines(
    path: Path,
    *,
    environment: str,
    source_id: str,
    partition: str,
) -> tuple[list[bytes], list[CanonicalRollout]]:
    lines: list[bytes] = []
    records: list[CanonicalRollout] = []
    with path.open("rb") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
                record = CanonicalRollout.model_validate(value)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"Invalid canonical record in {path} at line {line_number}: {error}"
                ) from error
            if (
                record.source.environment != environment
                or record.source.source_id != source_id
            ):
                continue
            lines.append(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
            records.append(record)
    if not records:
        raise ValueError(
            f"No {partition} records matched environment={environment!r}, "
            f"source_id={source_id!r}."
        )
    return lines, records


def _percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _token_summary(records: Sequence[CanonicalRollout]) -> JsonObject:
    lengths: list[int] = []
    supervised: list[int] = []
    for record in records:
        prepared = record.attributes.get("prepared_tokenization")
        if not isinstance(prepared, Mapping):
            raise ValueError(
                f"Record {record.id} has no prepared tokenization metadata."
            )
        tokens = prepared.get("tokens")
        target_tokens = prepared.get("supervised_tokens")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
            or isinstance(target_tokens, bool)
            or not isinstance(target_tokens, int)
            or target_tokens <= 0
        ):
            raise ValueError(f"Record {record.id} has invalid prepared token counts.")
        lengths.append(tokens)
        supervised.append(target_tokens)
    return {
        "records": len(records),
        "tokens": {
            "records": len(records),
            "min": min(lengths),
            "p50": _percentile(lengths, 0.50),
            "p75": _percentile(lengths, 0.75),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths),
            "total": sum(lengths),
        },
        "supervised_tokens": {
            "records": len(records),
            "min": min(supervised),
            "p50": _percentile(supervised, 0.50),
            "p95": _percentile(supervised, 0.95),
            "max": max(supervised),
            "total": sum(supervised),
        },
    }


def _count_records(records: Sequence[CanonicalRollout], field: str) -> dict[str, int]:
    if field == "teacher":
        values = (record.source.teacher for record in records)
    elif field == "environment":
        values = (record.source.environment for record in records)
    elif field == "category":
        values = (
            value
            if isinstance(value := record.attributes.get("category"), str)
            else "uncategorized"
            for record in records
        )
    else:
        raise ValueError(f"Unsupported count field: {field}")
    return dict(sorted(Counter(values).items()))


def _normalization_summary(records: Sequence[CanonicalRollout]) -> JsonObject:
    traces = messages = tool_calls = 0
    for record in records:
        value = record.attributes.get("parallel_spawn_normalization")
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"Record {record.id} has invalid normalization metadata.")
        traces += 1
        messages += int(value.get("messages", 0))
        tool_calls += int(value.get("tool_calls", 0))
    return {
        "strategy": "parallel_spawn_calls_to_single_call_turns",
        "traces": traces,
        "messages": messages,
        "tool_calls": tool_calls,
    }


def _write_bytes(path: Path, lines: Sequence[bytes]) -> None:
    with path.open("wb") as file:
        for line in lines:
            file.write(line)
        file.flush()
        os.fsync(file.fileno())


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def derive_source_view(
    parent_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str,
    dataset_version: str,
    environment: str,
    source_id: str,
    expected_train_records: int | None = None,
    expected_validation_records: int | None = None,
) -> DerivedSourceView:
    """Filter one source from an immutable release without certifying a new build."""
    parent_dir = Path(parent_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Derived dataset view already exists: {output_dir}")
    parent_manifest_path = parent_dir / "manifest.json"
    parent_train_path = parent_dir / "train.jsonl"
    parent_validation_path = parent_dir / "validation.jsonl"
    parent = _read_manifest(parent_manifest_path)
    _validate_prepared_file(parent, parent_train_path, filename="train.jsonl")
    _validate_prepared_file(parent, parent_validation_path, filename="validation.jsonl")

    train_lines, train_records = _read_selected_lines(
        parent_train_path,
        environment=environment,
        source_id=source_id,
        partition="train",
    )
    validation_lines, validation_records = _read_selected_lines(
        parent_validation_path,
        environment=environment,
        source_id=source_id,
        partition="validation",
    )
    if (
        expected_train_records is not None
        and len(train_records) != expected_train_records
    ):
        raise ValueError(
            f"Expected {expected_train_records} train records, "
            f"found {len(train_records)}."
        )
    if (
        expected_validation_records is not None
        and len(validation_records) != expected_validation_records
    ):
        raise ValueError(
            "Expected "
            f"{expected_validation_records} validation records, "
            f"found {len(validation_records)}."
        )
    all_records = [*train_records, *validation_records]
    ids = [record.id for record in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError("The derived source view contains duplicate rollout IDs.")
    train_groups = sorted({record.group_id for record in train_records})
    validation_groups = sorted({record.group_id for record in validation_records})
    overlap = sorted(set(train_groups) & set(validation_groups))
    if overlap:
        raise ValueError(
            "The derived source view leaks groups across splits: " + ", ".join(overlap)
        )

    source_entries = [
        deepcopy(source)
        for source in parent.get("sources", [])
        if isinstance(source, Mapping) and source.get("id") == source_id
    ]
    if len(source_entries) != 1:
        raise ValueError(
            f"Expected one parent source-manifest entry for {source_id!r}, "
            f"found {len(source_entries)}."
        )
    parent_identity = parent["dataset"]
    parent_fingerprint = str(parent_identity["fingerprint"])
    selector = {"environment": environment, "source_id": source_id}
    logical_build_spec = {
        "derivation": "single_source_candidate_view_v1",
        "parent": {
            "id": parent_identity["id"],
            "version": parent_identity["version"],
            "fingerprint": parent_fingerprint,
            "manifest_sha256": sha256_file(parent_manifest_path),
        },
        "selector": selector,
        "split_policy": "preserve_parent_assignments",
    }
    tokenization = deepcopy(parent.get("tokenization"))
    if isinstance(tokenization, dict):
        train_token_stats = _token_summary(train_records)
        validation_token_stats = _token_summary(validation_records)
        tokenization["splits"] = {
            "train": {
                "derivation": "inherited_parent_token_filter",
                "before_filter": train_token_stats["tokens"],
                "after_filter": train_token_stats["tokens"],
                "excluded": 0,
                "excluded_records": [],
                "supervised_tokens": train_token_stats["supervised_tokens"],
            },
            "validation": {
                "derivation": "inherited_parent_token_filter",
                "before_filter": validation_token_stats["tokens"],
                "after_filter": validation_token_stats["tokens"],
                "excluded": 0,
                "excluded_records": [],
                "supervised_tokens": validation_token_stats["supervised_tokens"],
            },
        }

    manifest: JsonObject = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "dataset": {"id": dataset_id, "version": dataset_version},
        "build_spec": {
            "version": 1,
            "sha256": sha256_text(canonical_json(logical_build_spec)),
            "config": logical_build_spec,
        },
        "preparation": {
            "git_revision": parent.get("preparation", {}).get("git_revision"),
            "candidate_view": True,
            "adapter_versions": parent.get("preparation", {}).get(
                "adapter_versions", {}
            ),
        },
        "policy": deepcopy(parent.get("policy")),
        "sources": source_entries,
        "filtering": {
            "strategy": "exact_environment_and_source_id",
            "selector": selector,
            "parent_records": parent.get("records", {}).get("total"),
            "excluded_other_sources": int(parent.get("records", {}).get("total", 0))
            - len(all_records),
            "included": len(all_records),
        },
        "normalization": _normalization_summary(all_records),
        "split": {
            "strategy": "preserve_parent_assignments",
            "parent_fingerprint": parent_fingerprint,
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
            "train_group_ids": train_groups,
            "validation_group_ids": validation_groups,
        },
        "records": {
            "total": len(all_records),
            "train": len(train_records),
            "validation": len(validation_records),
            "train_by_teacher": _count_records(train_records, "teacher"),
            "validation_by_teacher": _count_records(validation_records, "teacher"),
            "train_by_environment": _count_records(train_records, "environment"),
            "validation_by_environment": _count_records(
                validation_records, "environment"
            ),
            "train_by_category": _count_records(train_records, "category"),
            "validation_by_category": _count_records(validation_records, "category"),
        },
        "content": deepcopy(parent.get("content")),
        "derivation": {
            "schema_version": 1,
            "candidate": True,
            "parent_dataset": deepcopy(parent_identity),
            "parent_manifest_sha256": sha256_file(parent_manifest_path),
            "parent_prepared_files": deepcopy(parent.get("prepared_files")),
            "selector": selector,
            "record_ids": {
                "train": [record.id for record in train_records],
                "validation": [record.id for record in validation_records],
                "ordered_sha256": sha256_text(canonical_json(ids)),
            },
        },
    }
    if tokenization is not None:
        manifest["tokenization"] = tokenization

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        train_path = temporary_dir / "train.jsonl"
        validation_path = temporary_dir / "validation.jsonl"
        _write_bytes(train_path, train_lines)
        _write_bytes(validation_path, validation_lines)
        manifest["prepared_files"] = {
            "train.jsonl": {
                "bytes": train_path.stat().st_size,
                "sha256": sha256_file(train_path),
            },
            "validation.jsonl": {
                "bytes": validation_path.stat().st_size,
                "sha256": sha256_file(validation_path),
            },
        }
        manifest["dataset"]["fingerprint"] = compute_dataset_fingerprint(manifest)
        _write_json(temporary_dir / "manifest.json", manifest)
        if output_dir.exists():
            raise FileExistsError(
                f"Derived dataset view was created concurrently: {output_dir}"
            )
        os.rename(temporary_dir, output_dir)
    except BaseException:
        if temporary_dir.exists():
            import shutil

            shutil.rmtree(temporary_dir)
        raise

    return DerivedSourceView(
        output_dir=output_dir,
        train_path=output_dir / "train.jsonl",
        validation_path=output_dir / "validation.jsonl",
        manifest_path=output_dir / "manifest.json",
        manifest=manifest,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--expected-train-records", type=int)
    parser.add_argument("--expected-validation-records", type=int)
    args = parser.parse_args(argv)
    derived = derive_source_view(
        args.parent_dir,
        args.output_dir,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        environment=args.environment,
        source_id=args.source_id,
        expected_train_records=args.expected_train_records,
        expected_validation_records=args.expected_validation_records,
    )
    print(
        json.dumps(
            {
                "dataset": derived.manifest["dataset"],
                "output_dir": str(derived.output_dir),
                "records": derived.manifest["records"],
                "split": derived.manifest["split"],
                "tokenization": derived.manifest.get("tokenization"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
