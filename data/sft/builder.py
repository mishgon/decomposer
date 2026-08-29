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

from decomposer.core import build_decomposer_chat_tools
from decomposer.prompts import resolve_decomposer_system_prompt

from .adapters import ADAPTER_VERSIONS, ADAPTERS
from .schema import (
    CANONICAL_SCHEMA_VERSION,
    EXCLUSION_REASONS,
    MANIFEST_FORMAT_VERSION,
    BuildSpec,
    CanonicalRollout,
    JsonObject,
    TokenizationSpec,
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
    """Load a strict spec and resolve native source paths relative to the spec."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset build specification does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Dataset build specification {path} must contain an object.")
    spec = BuildSpec.model_validate(raw)

    def resolve(candidate: Path | None) -> Path | None:
        if candidate is None:
            return None
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (path.parent / candidate).resolve()
        )

    resolved_sources = []
    for source in spec.sources:
        gaia2 = source.gaia2
        if gaia2 is not None:
            gaia2 = gaia2.model_copy(
                update={"split_manifest": resolve(gaia2.split_manifest)}
            )
        resolved_sources.append(
            source.model_copy(update={"path": resolve(source.path), "gaia2": gaia2})
        )
    split = spec.split
    if split.manifest is not None:
        split = split.model_copy(update={"manifest": resolve(split.manifest)})
    return LoadedBuildSpec(
        path=path,
        sha256=sha256_file(path),
        spec=spec.model_copy(
            update={"sources": tuple(resolved_sources), "split": split}
        ),
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


def _parallel_spawn_normalization_counts(
    records: Sequence[CanonicalRollout],
) -> JsonObject:
    traces = 0
    messages = 0
    tool_calls = 0
    for record in records:
        value = record.attributes.get("parallel_spawn_normalization")
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Rollout {record.id} has invalid parallel-spawn normalization metadata."
            )
        record_messages = value.get("messages")
        record_tool_calls = value.get("tool_calls")
        if (
            not isinstance(record_messages, int)
            or isinstance(record_messages, bool)
            or record_messages <= 0
            or not isinstance(record_tool_calls, int)
            or isinstance(record_tool_calls, bool)
            or record_tool_calls < 2 * record_messages
        ):
            raise ValueError(
                f"Rollout {record.id} has invalid parallel-spawn normalization counts."
            )
        traces += 1
        messages += record_messages
        tool_calls += record_tool_calls
    return {"traces": traces, "messages": messages, "tool_calls": tool_calls}


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


def _allocate_pinned_split(
    records: Sequence[CanonicalRollout],
    *,
    manifest_path: Path,
    seed: int,
) -> tuple[list[CanonicalRollout], list[CanonicalRollout], JsonObject]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pinned split manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid pinned split manifest: {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("Pinned split manifest must contain an object")
    expected_keys = {"schema_version", "name", "seed", "parents", "groups", "summary"}
    if set(manifest) != expected_keys or manifest.get("schema_version") != 1:
        raise ValueError("Pinned split manifest has an unsupported schema")
    name = manifest.get("name")
    parents = manifest.get("parents")
    groups = manifest.get("groups")
    summary = manifest.get("summary")
    if (
        not isinstance(name, str)
        or not name
        or manifest.get("seed") != seed
        or not isinstance(parents, Mapping)
        or not isinstance(groups, list)
        or not groups
        or not isinstance(summary, Mapping)
    ):
        raise ValueError("Pinned split manifest metadata is invalid")

    assignments: dict[str, tuple[str, str, str]] = {}
    partition_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for index, raw_group in enumerate(groups):
        if not isinstance(raw_group, Mapping) or set(raw_group) != {
            "category",
            "group_id",
            "partition",
            "source",
        }:
            raise ValueError(f"Pinned split group {index} is invalid")
        group_id = raw_group.get("group_id")
        category = raw_group.get("category")
        partition = raw_group.get("partition")
        source = raw_group.get("source")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in assignments
            or not isinstance(category, str)
            or not category
            or partition not in {"train", "validation"}
            or not isinstance(source, str)
            or not source
        ):
            raise ValueError(f"Pinned split group {index} is invalid or duplicated")
        assignments[group_id] = (partition, category, source)
        partition_counts[partition] += 1
        source_counts[source] += 1

    expected_summary = {
        "groups": len(assignments),
        "train_groups": partition_counts["train"],
        "validation_groups": partition_counts["validation"],
        "groups_by_source": dict(sorted(source_counts.items())),
    }
    if dict(summary) != expected_summary:
        raise ValueError("Pinned split manifest summary changed")

    materialized_groups: set[str] = set()
    train: list[CanonicalRollout] = []
    validation: list[CanonicalRollout] = []
    for record in records:
        assignment = assignments.get(record.group_id)
        if assignment is None:
            raise ValueError(
                f"Task group {record.group_id!r} is missing from the pinned split"
            )
        partition, category, _source = assignment
        record_category = _record_category(record)
        if record_category != category:
            raise ValueError(
                f"Task group {record.group_id!r} changed category from "
                f"{category!r} to {record_category!r}"
            )
        materialized_groups.add(record.group_id)
        (train if partition == "train" else validation).append(record)
    if not train or not validation:
        raise ValueError("Pinned split must materialize non-empty train and validation")

    missing_groups = sorted(set(assignments) - materialized_groups)
    return (
        train,
        validation,
        {
            "strategy": "pinned",
            "group_key": "adapter-supplied stable task group ID",
            "seed": seed,
            "assignment_manifest": {
                "name": name,
                "sha256": sha256_file(manifest_path),
                "parents": dict(parents),
            },
            "num_groups": len(assignments),
            "train_groups": partition_counts["train"],
            "validation_groups": partition_counts["validation"],
            "materialized_groups": len(materialized_groups),
            "unmaterialized_groups": len(missing_groups),
            "unmaterialized_group_ids": missing_groups,
            "validation_group_ids": sorted(
                group_id
                for group_id, (partition, _category, _source) in assignments.items()
                if partition == "validation"
            ),
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


def _summarize_lengths(values: Sequence[int]) -> JsonObject:
    if not values:
        raise ValueError("Cannot summarize an empty token-length collection.")
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "records": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "total": sum(ordered),
    }


def _load_tokenization_runtime(
    spec: TokenizationSpec,
) -> tuple[Any, str, JsonObject]:
    from transformers import AutoTokenizer

    from training.sft.model_support import build_training_template

    tokenizer = AutoTokenizer.from_pretrained(
        spec.tokenizer,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
    )
    canonical_template = tokenizer.chat_template
    training_template = build_training_template(spec.profile, canonical_template)
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    resolved_revision = (
        init_kwargs.get("_commit_hash") if isinstance(init_kwargs, Mapping) else None
    )
    return (
        tokenizer,
        training_template,
        {
            "profile": spec.profile,
            "tokenizer": spec.tokenizer,
            "requested_revision": spec.revision,
            "resolved_revision": resolved_revision,
            "tokenizer_class": type(tokenizer).__name__,
            "canonical_template_sha256": sha256_text(canonical_template),
            "training_template_sha256": sha256_text(training_template),
            "include_reasoning": False,
            "max_tokens": spec.max_tokens,
        },
    )


def _tokenize_and_filter_split(
    records: Sequence[CanonicalRollout],
    *,
    split: str,
    spec: TokenizationSpec,
    tokenizer: Any,
    training_template: str,
) -> tuple[list[CanonicalRollout], JsonObject]:
    from training.sft.preprocessing import (
        PREPARED_TOKENIZATION_ATTRIBUTE,
        configure_example,
        tokenization_stats,
    )

    retained: list[CanonicalRollout] = []
    excluded: list[JsonObject] = []
    raw_lengths: list[int] = []
    retained_lengths: list[int] = []
    for record in records:
        serialized = record.model_dump(mode="json")
        configured = configure_example(serialized, include_reasoning=False)
        stats = tokenization_stats(
            {**serialized, **configured},
            tokenizer=tokenizer,
            training_template=training_template,
        )
        token_length = int(stats["_token_length"])
        supervised_tokens = int(stats["_supervised_tokens"])
        raw_lengths.append(token_length)
        prepared_metadata = {
            "profile": spec.profile,
            "tokens": token_length,
            "supervised_tokens": supervised_tokens,
        }
        enriched = record.model_copy(
            update={
                "attributes": {
                    **record.attributes,
                    PREPARED_TOKENIZATION_ATTRIBUTE: prepared_metadata,
                }
            }
        )
        if token_length <= spec.max_tokens:
            retained.append(enriched)
            retained_lengths.append(token_length)
        else:
            excluded.append(
                {
                    "id": record.id,
                    "split": split,
                    "source_id": record.source.source_id,
                    "environment": record.source.environment,
                    "token_length": token_length,
                    "max_tokens": spec.max_tokens,
                }
            )
    if not retained:
        raise ValueError(
            f"The {spec.max_tokens}-token preparation limit emptied the {split} split."
        )
    return retained, {
        "before_filter": _summarize_lengths(raw_lengths),
        "after_filter": _summarize_lengths(retained_lengths),
        "excluded": len(excluded),
        "excluded_records": excluded,
    }


def _logical_spec(spec: BuildSpec) -> JsonObject:
    policy_exclude = {"subagent_types"} if spec.spec_version == 1 else set()
    source_exclude = {"path"}
    if spec.spec_version < 3:
        source_exclude.add("selection")
    if spec.spec_version == 1:
        source_exclude.update(
            {
                "trace_format",
                "subagent_type_aliases",
                "sampling",
                "gaia2",
                "expected_native_rollouts",
                "expected_candidates",
                "require_completed_run",
            }
        )
    logical_sources = []
    for source in spec.sources:
        value = source.model_dump(mode="json", exclude=source_exclude)
        gaia2 = value.get("gaia2")
        if isinstance(gaia2, dict):
            gaia2.pop("split_manifest", None)
        else:
            value.pop("gaia2", None)
        logical_sources.append(value)
    logical_split = spec.split.model_dump(mode="json", exclude_none=True)
    logical_split.pop("manifest", None)
    logical = {
        "spec_version": spec.spec_version,
        "dataset": spec.dataset.model_dump(mode="json"),
        "policy": spec.policy.model_dump(mode="json", exclude=policy_exclude),
        "sources": logical_sources,
        "selection": spec.selection.model_dump(mode="json"),
        "split": logical_split,
    }
    if spec.tokenization is not None:
        logical["tokenization"] = spec.tokenization.model_dump(mode="json")
    return logical


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
    source_paths: Mapping[str, str | Path] | None = None,
) -> PreparedDataset:
    """Build one immutable dataset release from a checked-in specification."""
    if not isinstance(loaded, LoadedBuildSpec):
        loaded = load_build_spec(loaded)
    spec = loaded.spec
    overrides = {
        source_id: Path(path).expanduser().resolve()
        for source_id, path in (source_paths or {}).items()
    }
    known_source_ids = {source.id for source in spec.sources}
    unknown_overrides = sorted(set(overrides) - known_source_ids)
    if unknown_overrides:
        raise ValueError(
            "Unknown source path override IDs: " + ", ".join(unknown_overrides)
        )
    resolved_sources = tuple(
        source.model_copy(update={"path": overrides.get(source.id, source.path)})
        for source in spec.sources
    )
    missing_paths = sorted(
        source.id for source in resolved_sources if source.path is None
    )
    if missing_paths:
        raise ValueError(
            "Dataset sources require explicit path overrides: "
            + ", ".join(missing_paths)
        )
    spec = spec.model_copy(update={"sources": resolved_sources})
    revision = git_revision or _git_revision(require_clean=require_clean_git)
    output_root = Path(output_root).resolve()
    release_dir = output_root / spec.dataset.id / spec.dataset.version
    if release_dir.exists():
        raise FileExistsError(
            f"Dataset release already exists and is immutable: {release_dir}"
        )

    system_prompt_profile = spec.policy.resolved_system_prompt_profile
    system_prompt = resolve_decomposer_system_prompt(system_prompt_profile)
    canonical_subagent_type_ids = frozenset(
        subagent.id for subagent in spec.policy.subagent_types
    )
    canonical_tools: list[JsonObject] | None = None
    if spec.policy.subagent_types:
        canonical_tools = build_decomposer_chat_tools(
            [
                {
                    "subagent_type_id": subagent.id,
                    "description": subagent.description,
                    "assistant_id": subagent.id,
                }
                for subagent in spec.policy.subagent_types
            ]
        )
    records: list[CanonicalRollout] = []
    source_manifests: list[JsonObject] = []
    counts_by_source: dict[str, Counter[str]] = {}
    seen_ids: set[str] = set()
    for source in spec.sources:
        adapter = ADAPTERS.get(source.adapter)
        if adapter is None:
            raise ValueError(f"Unsupported dataset adapter: {source.adapter}")
        source_selection = spec.selection
        if source.selection is not None:
            source_selection = spec.selection.model_copy(
                update=source.selection.model_dump()
            )
        result = adapter(
            source,
            source_selection,
            system_prompt=system_prompt,
            canonical_tools=canonical_tools,
            canonical_subagent_type_ids=canonical_subagent_type_ids,
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
        source_records = [
            record for record in retained if record.source.source_id == source_id
        ]
        counts["excluded_prompt_teacher_cap"] += cap_exclusions[source_id]
        counts["included"] = len(source_records)
        _assert_filter_counts(counts, source_id)
        source_manifest["counts"] = _serialized_counts(counts)
        source_manifest["normalization"] = _parallel_spawn_normalization_counts(
            source_records
        )
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
    elif spec.split.strategy == "pinned":
        if spec.split.manifest is None:
            raise AssertionError("Pinned split manifest was not resolved")
        train_records, validation_records, split_manifest = _allocate_pinned_split(
            retained,
            manifest_path=spec.split.manifest,
            seed=spec.split.seed,
        )
    else:
        train_records, validation_records, split_manifest = _preserve_source_split(
            retained
        )
    train_records = _sort_records(train_records)
    validation_records = _sort_records(validation_records)

    tokenization_manifest: JsonObject | None = None
    if spec.tokenization is not None:
        tokenizer, training_template, tokenization_manifest = (
            _load_tokenization_runtime(spec.tokenization)
        )
        train_records, train_tokenization = _tokenize_and_filter_split(
            train_records,
            split="train",
            spec=spec.tokenization,
            tokenizer=tokenizer,
            training_template=training_template,
        )
        validation_records, validation_tokenization = _tokenize_and_filter_split(
            validation_records,
            split="validation",
            spec=spec.tokenization,
            tokenizer=tokenizer,
            training_template=training_template,
        )
        tokenization_manifest["splits"] = {
            "train": train_tokenization,
            "validation": validation_tokenization,
        }
    retained = [*train_records, *validation_records]
    split_manifest["effective_train_groups"] = len(
        {record.group_id for record in train_records}
    )
    split_manifest["effective_validation_groups"] = len(
        {record.group_id for record in validation_records}
    )

    total_counts = _empty_counts()
    for counts in counts_by_source.values():
        total_counts.update(counts)
    _assert_filter_counts(total_counts, "all sources")
    excluded_token_length = (
        0
        if tokenization_manifest is None
        else sum(
            int(split["excluded"]) for split in tokenization_manifest["splits"].values()
        )
    )
    excluded_token_length_by_source: Counter[str] = Counter()
    if tokenization_manifest is not None:
        for split in tokenization_manifest["splits"].values():
            for excluded in split["excluded_records"]:
                excluded_token_length_by_source[str(excluded["source_id"])] += 1
    excluded_malformed_by_source = {
        source_id: sum(
            counts[reason]
            for reason in EXCLUSION_REASONS
            if reason not in _SELECTION_EXCLUSION_REASONS
        )
        for source_id, counts in sorted(counts_by_source.items())
    }
    if tokenization_manifest is not None:
        for source_manifest in source_manifests:
            source_id = str(source_manifest["id"])
            before_token_limit = int(source_manifest["counts"]["included"])
            excluded = excluded_token_length_by_source[source_id]
            source_manifest["tokenization"] = {
                "eligible_before_token_limit": before_token_limit,
                "excluded_token_length": excluded,
                "included": before_token_limit - excluded,
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
            "subagent_types": [
                subagent.model_dump(mode="json")
                for subagent in spec.policy.subagent_types
            ],
        },
        "sources": source_manifests,
        "filtering": {
            **_serialized_counts(total_counts),
            "excluded_malformed_by_source": excluded_malformed_by_source,
            "eligible_before_token_limit": total_counts["included"],
            "excluded_token_length": excluded_token_length,
            "excluded_token_length_by_source": dict(
                sorted(excluded_token_length_by_source.items())
            ),
            "included": len(retained),
            "sidecar_failure_records": sum(
                int(source["sidecar_failure_records"]) for source in source_manifests
            ),
        },
        "normalization": {
            "strategy": "parallel_spawn_calls_to_single_call_turns",
            **_parallel_spawn_normalization_counts(retained),
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
    if tokenization_manifest is not None:
        manifest["tokenization"] = tokenization_manifest

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
