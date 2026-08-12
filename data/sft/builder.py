"""Deterministic construction of immutable canonical SFT dataset releases."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from decomposer.prompts import DECOMPOSER_SYSTEM_PROMPT

from .adapters import ADAPTERS, ADAPTER_VERSIONS
from .schema import (
    CANONICAL_SCHEMA_VERSION,
    EXCLUSION_REASONS,
    MANIFEST_FORMAT_VERSION,
    BuildSpec,
    CanonicalRollout,
    JsonObject,
    canonical_json,
    sha256_file,
    sha256_text,
)


@dataclass(frozen=True)
class LoadedBuildSpec:
    path: Path
    sha256: str
    spec: BuildSpec


@dataclass(frozen=True)
class PreparedDataset:
    release_dir: Path
    train_path: Path
    validation_path: Path
    manifest_path: Path
    manifest: JsonObject


def load_build_spec(path: str | Path) -> LoadedBuildSpec:
    """Load a strict spec and resolve native source paths relative to the spec."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset build specification does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Dataset build specification {path} must contain an object.")
    spec = BuildSpec.model_validate(raw)
    resolved_sources = tuple(
        source.model_copy(
            update={
                "path": (
                    source.path.resolve()
                    if source.path.is_absolute()
                    else (path.parent / source.path).resolve()
                )
            }
        )
        for source in spec.sources
    )
    return LoadedBuildSpec(
        path=path,
        sha256=sha256_file(path),
        spec=spec.model_copy(update={"sources": resolved_sources}),
    )


def _git_revision(*, require_clean: bool) -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Dataset releases must be built from a Git checkout."
        ) from error
    if require_clean and status:
        raise RuntimeError(
            "Dataset releases require a clean tracked worktree; commit the preparation "
            "implementation and build specification first."
        )
    return revision


def _empty_counts() -> Counter[str]:
    return Counter({reason: 0 for reason in EXCLUSION_REASONS})


def _serialized_counts(counts: Counter[str]) -> JsonObject:
    return {
        "rollouts": counts["rollouts"],
        "eligible_before_cap": counts["eligible"],
        "included": counts["included"],
        **{reason: counts[reason] for reason in EXCLUSION_REASONS},
    }


def _assert_filter_counts(counts: Counter[str], description: str) -> None:
    invalid = sum(
        counts[reason]
        for reason in EXCLUSION_REASONS
        if reason not in {"excluded_reward", "excluded_prompt_teacher_cap"}
    )
    if counts["rollouts"] != counts["excluded_reward"] + invalid + counts["eligible"]:
        raise AssertionError(
            f"Rollout filtering counts do not add up for {description}."
        )
    if counts["eligible"] != counts["included"] + counts["excluded_prompt_teacher_cap"]:
        raise AssertionError(
            f"Prompt-cap filtering counts do not add up for {description}."
        )


def _apply_prompt_teacher_cap(
    records: Sequence[CanonicalRollout],
    *,
    limit: int | None,
    seed: int,
) -> tuple[list[CanonicalRollout], Counter[str]]:
    exclusions: Counter[str] = Counter()
    if limit is None:
        return list(records), exclusions
    groups: dict[tuple[str, str], list[CanonicalRollout]] = defaultdict(list)
    for record in records:
        groups[(record.group_id, record.source.teacher)].append(record)
    retained: list[CanonicalRollout] = []
    for group in groups.values():
        ranked = sorted(group, key=lambda record: sha256_text(f"{seed}\0{record.id}"))
        retained.extend(ranked[:limit])
        for excluded in ranked[limit:]:
            exclusions[excluded.source.source_id] += 1
    return retained, exclusions


def _record_category(record: CanonicalRollout) -> str:
    category = record.attributes.get("category")
    return category if isinstance(category, str) and category else "uncategorized"


def _allocate_prompt_fixed_split(
    records: Sequence[CanonicalRollout],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[CanonicalRollout], list[CanonicalRollout], JsonObject]:
    group_categories: dict[str, str] = {}
    groups_by_category: dict[str, set[str]] = defaultdict(set)
    for record in records:
        category = _record_category(record)
        existing = group_categories.setdefault(record.group_id, category)
        if existing != category:
            raise ValueError(
                f"Task group {record.group_id} spans categories {existing!r} and {category!r}."
            )
        groups_by_category[category].add(record.group_id)
    num_groups = len(group_categories)
    target = round(num_groups * validation_fraction)
    raw = {
        category: len(groups) * target / num_groups
        for category, groups in groups_by_category.items()
    }
    quotas = {category: math.floor(value) for category, value in raw.items()}
    remaining = target - sum(quotas.values())
    remainders = sorted(
        (
            (raw[category] - quotas[category], category)
            for category in groups_by_category
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for _, category in remainders[:remaining]:
        quotas[category] += 1
    validation_groups: set[str] = set()
    for category, groups in groups_by_category.items():
        ranked = sorted(groups, key=lambda key: sha256_text(f"{seed}\0{key}"))
        validation_groups.update(ranked[: quotas[category]])
    train = [record for record in records if record.group_id not in validation_groups]
    validation = [record for record in records if record.group_id in validation_groups]
    return (
        train,
        validation,
        {
            "strategy": "prompt_fixed",
            "group_key": "adapter-supplied stable task group ID",
            "seed": seed,
            "validation_fraction": validation_fraction,
            "num_groups": num_groups,
            "train_groups": num_groups - len(validation_groups),
            "validation_groups": len(validation_groups),
            "validation_groups_by_category": dict(sorted(quotas.items())),
            "validation_group_ids": sorted(validation_groups),
        },
    )


def _preserve_source_split(
    records: Sequence[CanonicalRollout],
) -> tuple[list[CanonicalRollout], list[CanonicalRollout], JsonObject]:
    train = [record for record in records if record.source.partition == "train"]
    validation = [
        record for record in records if record.source.partition == "validation"
    ]
    train_groups = {record.group_id for record in train}
    validation_groups = {record.group_id for record in validation}
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(
            "Source partitions leak task groups across train and validation: "
            + ", ".join(sorted(overlap)[:5])
        )
    return (
        train,
        validation,
        {
            "strategy": "preserve",
            "group_key": "adapter-supplied stable task group ID",
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
        },
    )


def _sort_records(records: Sequence[CanonicalRollout]) -> list[CanonicalRollout]:
    return sorted(
        records,
        key=lambda record: (
            record.group_id,
            record.source.teacher,
            record.source.source_id,
            record.source.task_id,
            record.source.rollout_id,
        ),
    )


def _count_by(records: Sequence[CanonicalRollout], field: str) -> dict[str, int]:
    if field == "teacher":
        values = (record.source.teacher for record in records)
    elif field == "environment":
        values = (record.source.environment for record in records)
    elif field == "category":
        values = (_record_category(record) for record in records)
    else:
        raise ValueError(f"Unsupported record count field: {field}")
    return dict(sorted(Counter(values).items()))


def _logical_spec(spec: BuildSpec) -> JsonObject:
    return {
        "spec_version": spec.spec_version,
        "dataset": spec.dataset.model_dump(mode="json"),
        "policy": spec.policy.model_dump(mode="json"),
        "sources": [
            source.model_dump(mode="json", exclude={"path"}) for source in spec.sources
        ],
        "selection": spec.selection.model_dump(mode="json"),
        "split": spec.split.model_dump(mode="json", exclude_none=True),
    }


def compute_dataset_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Compute the portable identity of a manifest-v3 dataset release."""
    payload = deepcopy(dict(manifest))
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Dataset manifest has no dataset identity object.")
    dataset.pop("fingerprint", None)
    build_spec = payload.get("build_spec")
    if not isinstance(build_spec, dict):
        raise ValueError("Dataset manifest has no build_spec object.")
    build_spec.pop("sha256", None)
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Dataset manifest has no sources list.")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Dataset manifest source entries must be objects.")
        source.pop("locator", None)
    return sha256_text(canonical_json(payload))


def _write_jsonl(path: Path, records: Sequence[CanonicalRollout]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        file.flush()
        os.fsync(file.fileno())


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def prepare_dataset(
    loaded: LoadedBuildSpec | str | Path,
    output_root: str | Path,
    *,
    git_revision: str | None = None,
    require_clean_git: bool = True,
) -> PreparedDataset:
    """Build one immutable dataset release from a checked-in specification."""
    if not isinstance(loaded, LoadedBuildSpec):
        loaded = load_build_spec(loaded)
    spec = loaded.spec
    revision = git_revision or _git_revision(require_clean=require_clean_git)
    output_root = Path(output_root).resolve()
    release_dir = output_root / spec.dataset.id / spec.dataset.version
    if release_dir.exists():
        raise FileExistsError(
            f"Dataset release already exists and is immutable: {release_dir}"
        )

    system_prompt = DECOMPOSER_SYSTEM_PROMPT
    records: list[CanonicalRollout] = []
    source_manifests: list[JsonObject] = []
    counts_by_source: dict[str, Counter[str]] = {}
    seen_ids: set[str] = set()
    for source in spec.sources:
        adapter = ADAPTERS.get(source.adapter)
        if adapter is None:
            raise ValueError(f"Unsupported dataset adapter: {source.adapter}")
        result = adapter(
            source,
            spec.selection,
            system_prompt=system_prompt,
        )
        for record in result.records:
            if record.id in seen_ids:
                raise ValueError(f"Duplicate canonical rollout ID: {record.id}")
            seen_ids.add(record.id)
        records.extend(result.records)
        source_manifests.append(result.source_manifest)
        counts_by_source[source.id] = result.counts

    train_candidates = [
        record for record in records if record.source.partition == "train"
    ]
    retained_train, cap_exclusions = _apply_prompt_teacher_cap(
        train_candidates,
        limit=spec.selection.max_traces_per_prompt_per_teacher,
        seed=spec.split.seed,
    )
    retained = [
        *retained_train,
        *(record for record in records if record.source.partition == "validation"),
    ]
    for source_manifest in source_manifests:
        source_id = str(source_manifest["id"])
        counts = counts_by_source[source_id]
        counts["excluded_prompt_teacher_cap"] += cap_exclusions[source_id]
        counts["included"] = sum(
            record.source.source_id == source_id for record in retained
        )
        _assert_filter_counts(counts, source_id)
        source_manifest["counts"] = _serialized_counts(counts)
    if not retained:
        raise ValueError("No usable rollout traces were found.")

    tool_schemas = {canonical_json(record.tools) for record in retained}
    if len(tool_schemas) != 1:
        raise ValueError(
            f"Expected one consistent tool schema, found {len(tool_schemas)}."
        )
    if spec.split.strategy == "prompt_fixed":
        train_records, validation_records, split_manifest = (
            _allocate_prompt_fixed_split(
                retained,
                validation_fraction=float(spec.split.validation_fraction),
                seed=spec.split.seed,
            )
        )
    else:
        train_records, validation_records, split_manifest = _preserve_source_split(
            retained
        )
    train_records = _sort_records(train_records)
    validation_records = _sort_records(validation_records)

    total_counts = _empty_counts()
    for counts in counts_by_source.values():
        total_counts.update(counts)
    _assert_filter_counts(total_counts, "all sources")

    manifest: JsonObject = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "dataset": {
            "id": spec.dataset.id,
            "version": spec.dataset.version,
        },
        "build_spec": {
            "version": spec.spec_version,
            "sha256": loaded.sha256,
            "config": _logical_spec(spec),
        },
        "preparation": {
            "git_revision": revision,
            "adapter_versions": {
                name: ADAPTER_VERSIONS[name]
                for name in sorted({source.adapter for source in spec.sources})
            },
        },
        "policy": {
            "id": spec.policy.id,
            "system_prompt_sha256": sha256_text(system_prompt),
        },
        "sources": source_manifests,
        "filtering": {
            **_serialized_counts(total_counts),
            "sidecar_failure_records": sum(
                int(source["sidecar_failure_records"]) for source in source_manifests
            ),
        },
        "split": split_manifest,
        "records": {
            "total": len(retained),
            "train": len(train_records),
            "validation": len(validation_records),
            "train_by_teacher": _count_by(train_records, "teacher"),
            "validation_by_teacher": _count_by(validation_records, "teacher"),
            "train_by_environment": _count_by(train_records, "environment"),
            "validation_by_environment": _count_by(validation_records, "environment"),
            "train_by_category": _count_by(train_records, "category"),
            "validation_by_category": _count_by(validation_records, "category"),
        },
        "content": {
            "tool_schema_sha256": sha256_text(next(iter(tool_schemas))),
            "assistant_reasoning_preserved_as_metadata": True,
        },
    }

    release_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{spec.dataset.version}.", dir=release_dir.parent)
    )
    try:
        temporary_train = temporary_dir / "train.jsonl"
        temporary_validation = temporary_dir / "validation.jsonl"
        _write_jsonl(temporary_train, train_records)
        _write_jsonl(temporary_validation, validation_records)
        manifest["prepared_files"] = {
            "train.jsonl": {
                "bytes": temporary_train.stat().st_size,
                "sha256": sha256_file(temporary_train),
            },
            "validation.jsonl": {
                "bytes": temporary_validation.stat().st_size,
                "sha256": sha256_file(temporary_validation),
            },
        }
        manifest["dataset"]["fingerprint"] = compute_dataset_fingerprint(manifest)
        _write_json(temporary_dir / "manifest.json", manifest)
        if release_dir.exists():
            raise FileExistsError(
                f"Dataset release was created concurrently: {release_dir}"
            )
        os.rename(temporary_dir, release_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    return PreparedDataset(
        release_dir=release_dir,
        train_path=release_dir / "train.jsonl",
        validation_path=release_dir / "validation.jsonl",
        manifest_path=release_dir / "manifest.json",
        manifest=manifest,
    )
