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

from decomposer.prompt_profiles import resolve_decomposer_system_prompt

from .adapters.base import empty_counts
from .adapters.registry import ADAPTER_VERSIONS, ADAPTERS
from .snapshots import (
    DEFAULT_SNAPSHOT_ROOT,
    load_snapshot,
    snapshot_directory,
)
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

_SELECTION_EXCLUSION_REASONS = frozenset(
    {
        "excluded_reward",
        "excluded_quality",
        "excluded_prompt_teacher_cap",
    }
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
    """Load a strict build specification."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset build specification does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Dataset build specification {path} must contain an object.")
    return LoadedBuildSpec(
        path=path,
        sha256=sha256_file(path),
        spec=BuildSpec.model_validate(raw),
    )


_REPOSITORY = Path(__file__).resolve().parents[1]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_revision(
    *,
    require_clean: bool,
    spec_path: Path | None = None,
    repository: Path = _REPOSITORY,
) -> str:
    try:
        revision = _git(repository, "rev-parse", "HEAD")
        status = _git(repository, "status", "--short", "--untracked-files=no")
        # A new adapter or spec that was never committed is not in the recorded revision.
        untracked = _git(
            repository, "ls-files", "--others", "--exclude-standard", "--", "sft", "src"
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Dataset releases must be built from a Git checkout."
        ) from error
    if not require_clean:
        return revision
    if status or untracked:
        raise RuntimeError(
            "Dataset releases require a clean worktree with no untracked files under "
            "sft/ or src/; commit the preparation implementation and build "
            "specification first."
        )
    if spec_path is not None:
        try:
            relative_spec = spec_path.resolve().relative_to(repository.resolve())
            _git(repository, "ls-files", "--error-unmatch", "--", str(relative_spec))
        except (ValueError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"Dataset build specification {spec_path} must be committed in this "
                "repository."
            ) from error
    return revision


def _serialized_counts(counts: Counter[str]) -> JsonObject:
    malformed = sum(
        counts[reason]
        for reason in EXCLUSION_REASONS
        if reason not in _SELECTION_EXCLUSION_REASONS
    )
    return {
        "rollouts": counts["rollouts"],
        "eligible_before_cap": counts["eligible"],
        "included": counts["included"],
        "excluded_malformed": malformed,
        **{reason: counts[reason] for reason in EXCLUSION_REASONS},
    }


def _assert_filter_counts(counts: Counter[str], description: str) -> None:
    invalid = sum(
        counts[reason]
        for reason in EXCLUSION_REASONS
        if reason not in _SELECTION_EXCLUSION_REASONS
    )
    if counts["rollouts"] != (
        counts["excluded_reward"]
        + counts["excluded_quality"]
        + invalid
        + counts["eligible"]
    ):
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
    categories_by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        categories_by_group[record.group_id].add(_record_category(record))
    # A prompt can appear under several categories (for example, a tau2 domain and
    # its independent `_dsh` implementation). Its group stays whole, so the prompt
    # never lands on both sides, and counts toward its first category.
    groups_by_category: dict[str, set[str]] = defaultdict(set)
    for group_id, categories in categories_by_group.items():
        groups_by_category[min(categories)].add(group_id)
    num_groups = len(categories_by_group)
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
            "multi_category_groups": sum(
                len(categories) > 1 for categories in categories_by_group.values()
            ),
            "train_groups": num_groups - len(validation_groups),
            "validation_groups": len(validation_groups),
            "validation_groups_by_category": dict(sorted(quotas.items())),
            "validation_group_ids": sorted(validation_groups),
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
        "sources": [source.model_dump(mode="json") for source in spec.sources],
        "selection": spec.selection.model_dump(mode="json"),
        "split": spec.split.model_dump(mode="json"),
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
    snapshot_root: str | Path = DEFAULT_SNAPSHOT_ROOT,
) -> PreparedDataset:
    """Build one immutable dataset release from a checked-in specification."""
    if not isinstance(loaded, LoadedBuildSpec):
        loaded = load_build_spec(loaded)
    spec = loaded.spec
    snapshot_root = Path(snapshot_root).resolve()
    source_dirs: dict[str, Path] = {}
    for source in spec.sources:
        # A snapshot is located by its digest and verified file by file before
        # its adapter reads it.
        source_dirs[source.id] = snapshot_directory(
            snapshot_root, source.adapter, source.snapshot.removeprefix("sha256:")
        )
        load_snapshot(source_dirs[source.id], source.snapshot, adapter=source.adapter)
    revision = git_revision or _git_revision(
        require_clean=require_clean_git, spec_path=loaded.path
    )
    output_root = Path(output_root).resolve()
    release_dir = output_root / spec.dataset.id / spec.dataset.version
    if release_dir.exists():
        raise FileExistsError(
            f"Dataset release already exists and is immutable: {release_dir}"
        )

    system_prompt_profile = spec.policy.resolved_system_prompt_profile
    system_prompt = resolve_decomposer_system_prompt(system_prompt_profile)
    records: list[CanonicalRollout] = []
    source_manifests: list[JsonObject] = []
    counts_by_source: dict[str, Counter[str]] = {}
    seen_ids: set[str] = set()
    for source in spec.sources:
        source_selection = spec.selection
        if source.selection is not None:
            source_selection = spec.selection.model_copy(
                update=source.selection.model_dump()
            )
        result = ADAPTERS[source.adapter](
            source,
            source_selection,
            source_dir=source_dirs[source.id],
            system_prompt=system_prompt,
        )
        for record in result.records:
            if record.id in seen_ids:
                raise ValueError(f"Duplicate canonical rollout ID: {record.id}")
            seen_ids.add(record.id)
        records.extend(result.records)
        result.source_manifest["snapshot"] = source.snapshot
        source_manifests.append(result.source_manifest)
        counts_by_source[source.id] = result.counts

    retained, cap_exclusions = _apply_prompt_teacher_cap(
        records,
        limit=spec.selection.max_traces_per_prompt_per_teacher,
        seed=spec.split.seed,
    )
    for source_manifest in source_manifests:
        source_id = str(source_manifest["id"])
        counts = counts_by_source[source_id]
        source_records = [
            record for record in retained if record.source.source_id == source_id
        ]
        counts["excluded_prompt_teacher_cap"] += cap_exclusions[source_id]
        counts["included"] = len(source_records)
        _assert_filter_counts(counts, source_id)
        source_manifest["counts"] = _serialized_counts(counts)
    if not retained:
        raise ValueError("No usable rollout traces were found.")

    # Sources may expose different native tool schemas (for example, each gym's
    # own agent types), but each source must use one.
    tool_schemas_by_source: defaultdict[str, set[str]] = defaultdict(set)
    for record in retained:
        tool_schemas_by_source[record.source.source_id].add(
            sha256_text(canonical_json(record.tools))
        )
    for source_manifest in source_manifests:
        source_id = str(source_manifest["id"])
        schemas = tool_schemas_by_source[source_id]
        if len(schemas) > 1:
            raise ValueError(
                f"Expected one consistent tool schema in source {source_id!r}, "
                f"found {len(schemas)}."
            )
        source_manifest["tool_schema_sha256"] = next(iter(schemas), None)
    tool_schema_sha256s = sorted(set().union(*tool_schemas_by_source.values()))
    train_records, validation_records, split_manifest = _allocate_prompt_fixed_split(
        retained,
        validation_fraction=spec.split.validation_fraction,
        seed=spec.split.seed,
    )
    train_records = _sort_records(train_records)
    validation_records = _sort_records(validation_records)

    split_manifest["effective_train_groups"] = len(
        {record.group_id for record in train_records}
    )
    split_manifest["effective_validation_groups"] = len(
        {record.group_id for record in validation_records}
    )

    total_counts = empty_counts()
    for counts in counts_by_source.values():
        total_counts.update(counts)
    _assert_filter_counts(total_counts, "all sources")
    excluded_malformed_by_source = {
        source_id: sum(
            counts[reason]
            for reason in EXCLUSION_REASONS
            if reason not in _SELECTION_EXCLUSION_REASONS
        )
        for source_id, counts in sorted(counts_by_source.items())
    }

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
            "system_prompt_profile": system_prompt_profile,
            "system_prompt_sha256": sha256_text(system_prompt),
        },
        "sources": source_manifests,
        "filtering": {
            **_serialized_counts(total_counts),
            "excluded_malformed_by_source": excluded_malformed_by_source,
            "included": len(retained),
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
            "tool_schema_sha256s": tool_schema_sha256s,
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
