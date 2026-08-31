from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate.utils import merge_fsdp_weights, save_fsdp_model
from datasets import Dataset, Features, Value
from datasets import Json as DatasetJson
from datasets import List as DatasetList
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    GenerationConfig,
)
from trl import SFTConfig, SFTTrainer

from data.sft.builder import compute_dataset_fingerprint
from data.sft.schema import (
    CANONICAL_SCHEMA_VERSION,
    MANIFEST_FORMAT_VERSION,
    sha256_text,
)
from decomposer.prompts import (
    DECOMPOSER_PROMPT_PROFILES,
    resolve_decomposer_system_prompt,
)

from .clearml_logging import (
    SeparatePlotsClearMLCallback,
    validate_weight_norm_interval,
)
from .liger import configure_liger_for_model, liger_compatible_loss_type
from .model_support import (
    build_training_template,
    tokenization_profile_for_model_config,
    validate_reasoning_policy,
)
from .preprocessing import (
    PREPARED_TOKENIZATION_ATTRIBUTE,
    PREPARED_TOKENIZATION_PROFILE,
)
from .preprocessing import (
    configure_example as _configure_example,
)
from .preprocessing import (
    tokenization_stats as _tokenization_stats,
)
from .qwen35_fast_runtime import (
    DEFAULT_BUNDLE_DIR,
    EXPECTED_RUNTIME,
    HF_FA2_IMPLEMENTATION,
    PROFILE_NAME as QWEN35_FAST_PROFILE,
    validate_runtime_bundle,
)

JsonObject = dict[str, Any]
_LAUNCHER_LOG_FILENAMES = frozenset({"console.log", "mlspace.log"})
_GEMMA4_REQUIRED_STOP_TOKENS = ("<turn|>", "<|tool_response>")
_PREPARED_SPLIT_FEATURES = Features(
    {
        "schema_version": Value("int64"),
        "id": Value("string"),
        "group_id": Value("string"),
        "messages": DatasetList(DatasetJson()),
        "tools": DatasetList(DatasetJson()),
        "source": DatasetJson(),
        "outcome": DatasetJson(),
        "attributes": DatasetJson(),
    }
)


def _load_yaml(path: Path) -> JsonObject:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Training config {path} must contain a YAML object.")
    return value


def _nested_mapping(config: Mapping[str, Any], key: str) -> JsonObject:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {key!r} must be an object.")
    return dict(value)


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty collection.")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summarize_tokenization(dataset: Dataset) -> JsonObject:
    lengths = list(dataset["_token_length"])
    supervised = list(dataset["_supervised_tokens"])
    return {
        "records": len(dataset),
        "tokens": {
            "min": min(lengths),
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths),
            "total": sum(lengths),
        },
        "supervised_tokens": {
            "min": min(supervised),
            "p50": _percentile(supervised, 0.50),
            "p95": _percentile(supervised, 0.95),
            "max": max(supervised),
            "total": sum(supervised),
        },
    }


def _callable_identity(value: Any) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    return repr(value)


def _file_runtime(path: str | os.PathLike[str] | None) -> JsonObject | None:
    if path is None:
        return None
    candidate = Path(path).resolve()
    if not candidate.is_file():
        return {"path": str(candidate), "exists": False}
    digest = hashlib.sha256()
    with candidate.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(candidate),
        "exists": True,
        "size_bytes": candidate.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _callable_runtime(value: Any) -> JsonObject | None:
    if value is None:
        return None
    module = inspect.getmodule(value)
    module_file = getattr(module, "__file__", None)
    return {
        "identity": _callable_identity(value),
        "module_file": _file_runtime(module_file),
    }


def _module_runtime(module_name: str) -> JsonObject | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    return {
        "name": module_name,
        "origin": _file_runtime(spec.origin),
    }


def _optional_distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _qwen35_linear_attention_runtime() -> JsonObject:
    """Describe the implementations selected by Qwen3.5's optional fast path."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    return {
        "packages": {
            distribution: _optional_distribution_version(distribution)
            for distribution in (
                "flash-linear-attention",
                "fla-core",
                "causal-conv1d",
            )
        },
        "fast_path_available": bool(modeling_qwen3_5.is_fast_path_available),
        "causal_conv1d": _callable_identity(modeling_qwen3_5.causal_conv1d_fn),
        "causal_conv1d_update": _callable_identity(
            modeling_qwen3_5.causal_conv1d_update
        ),
        "chunk_gated_delta_rule": _callable_identity(
            modeling_qwen3_5.chunk_gated_delta_rule
        ),
        "fused_recurrent_gated_delta_rule": _callable_identity(
            modeling_qwen3_5.fused_recurrent_gated_delta_rule
        ),
        "gated_rms_norm": _callable_identity(modeling_qwen3_5.FusedRMSNormGated),
    }


def _attention_backend_runtime(
    requested_implementation: str,
    *,
    resolved_implementation: str | None = None,
) -> JsonObject:
    """Record the actual SDPA, Hub-kernel, or native FlashAttention callables."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    runtime: JsonObject = {
        "requested_implementation": requested_implementation,
        "resolved_implementation": resolved_implementation,
        "packages": {
            distribution: _optional_distribution_version(distribution)
            for distribution in (
                "flash-attn",
                "kernels",
                "kernels-data",
                "torch",
                "transformers",
            )
        },
        "modules": {
            module: _module_runtime(module)
            for module in (
                "flash_attn",
                "flash_attn_2_cuda",
                "kernels",
            )
        },
    }
    if "flash" not in requested_implementation:
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            requested_implementation,
            lambda *args, **kwargs: None,
        )
        runtime["attention_interface"] = _callable_runtime(interface)
        return runtime

    from transformers.integrations.flash_attention import flash_attention_forward
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

    functions, process_kwargs = lazy_import_flash_attention(
        requested_implementation,
        attention_wrapper=flash_attention_forward,
    )
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        requested_implementation,
        flash_attention_forward,
    )
    runtime["attention_interface"] = _callable_runtime(interface)
    names = (
        "flash_attn_func",
        "flash_attn_varlen_func",
        "flash_attn_with_kvcache",
        "pad_input",
        "unpad_input",
    )
    runtime["resolved_callables"] = {
        name: _callable_runtime(value) for name, value in zip(names, functions)
    }
    runtime["process_kwargs"] = _callable_runtime(process_kwargs)
    return runtime


def _validate_required_runtime_profile(
    run_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> JsonObject | None:
    """Fail closed when an accelerated recipe did not load its pinned kernels."""
    profile = run_config.get("required_runtime_profile")
    if profile is None:
        return None
    if profile != QWEN35_FAST_PROFILE:
        raise ValueError(f"Unsupported run.required_runtime_profile: {profile!r}.")
    launcher_profile = os.environ.get("DECOMPOSER_SFT_RUNTIME_PROFILE")
    if launcher_profile is not None and launcher_profile != profile:
        raise RuntimeError(
            f"Launcher runtime profile is {launcher_profile!r}, expected {profile!r}."
        )
    if os.environ.get("FLA_TILELANG") != "0":
        raise RuntimeError(
            "The Qwen3.5 fast runtime requires FLA_TILELANG=0 to select the "
            "validated Triton backend."
        )

    bundle_dir = Path(
        os.environ.get("DECOMPOSER_SFT_RUNTIME_BUNDLE", str(DEFAULT_BUNDLE_DIR))
    ).resolve()
    bundle = validate_runtime_bundle(bundle_dir, verify_checksums=False)
    if str(model_config.get("attn_implementation")) != HF_FA2_IMPLEMENTATION:
        raise RuntimeError(
            "The Qwen3.5 fast runtime requires the pinned HF FlashAttention-2 "
            f"implementation {HF_FA2_IMPLEMENTATION!r}."
        )
    if torch.__version__ != EXPECTED_RUNTIME["torch"]:
        raise RuntimeError(
            f"Qwen3.5 fast runtime requires torch {EXPECTED_RUNTIME['torch']}, "
            f"found {torch.__version__}."
        )
    if torch.version.cuda != EXPECTED_RUNTIME["torch_cuda"]:
        raise RuntimeError(
            f"Qwen3.5 fast runtime requires CUDA {EXPECTED_RUNTIME['torch_cuda']}, "
            f"found {torch.version.cuda}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The Qwen3.5 fast runtime requires an H100 CUDA device.")
    capability = ".".join(str(value) for value in torch.cuda.get_device_capability())
    if capability != EXPECTED_RUNTIME["gpu_arch"]:
        raise RuntimeError(
            f"Qwen3.5 fast runtime requires compute capability "
            f"{EXPECTED_RUNTIME['gpu_arch']}, found {capability}."
        )

    expected_packages = EXPECTED_RUNTIME["packages"]
    actual_packages = {
        name: _optional_distribution_version(name) for name in expected_packages
    }
    mismatches = {
        name: {"expected": expected, "actual": actual_packages[name]}
        for name, expected in expected_packages.items()
        if actual_packages[name] != expected
    }
    if mismatches:
        raise RuntimeError(
            "Qwen3.5 fast runtime package mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    linear_attention = _qwen35_linear_attention_runtime()
    required_linear_identities = {
        "causal_conv1d": "causal_conv1d",
        "causal_conv1d_update": "causal_conv1d",
        "chunk_gated_delta_rule": "fla.",
        "fused_recurrent_gated_delta_rule": "fla.",
        "gated_rms_norm": "fla.",
    }
    invalid_linear = {
        key: linear_attention.get(key)
        for key, fragment in required_linear_identities.items()
        if not isinstance(linear_attention.get(key), str)
        or fragment not in str(linear_attention[key])
    }
    if not linear_attention.get("fast_path_available") or invalid_linear:
        raise RuntimeError(
            "Transformers did not bind the complete Qwen3.5 FLA/causal fast path: "
            + json.dumps(invalid_linear, sort_keys=True)
        )

    attention = _attention_backend_runtime(HF_FA2_IMPLEMENTATION)
    resolved_callables = attention.get("resolved_callables")
    flash_identity = None
    if isinstance(resolved_callables, Mapping):
        flash = resolved_callables.get("flash_attn_func")
        if isinstance(flash, Mapping):
            flash_identity = flash.get("identity")
    if not isinstance(flash_identity, str) or "_flash_attn2_cuda_" not in flash_identity:
        raise RuntimeError(
            "Transformers did not load the pinned HF FlashAttention-2 CUDA callable; "
            f"resolved {flash_identity!r}."
        )
    return {
        "profile": profile,
        "bundle_dir": str(bundle_dir),
        "bundle": bundle,
        "packages": actual_packages,
        "linear_attention": linear_attention,
        "full_attention": attention,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": capability,
    }


def _limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("Dataset sample limits must be positive when specified.")
    return dataset.select(range(min(limit, len(dataset))))


def _select_longest_by_token_length(
    dataset: Dataset,
    limit: int | None,
) -> Dataset:
    """Select the longest examples after full tokenization and length filtering."""
    if limit is None:
        return dataset
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("data.longest_train_samples must be a positive integer.")
    indices = sorted(
        range(len(dataset)),
        key=lambda index: (-int(dataset[index]["_token_length"]), index),
    )
    return dataset.select(indices[: min(limit, len(indices))])


def _select_stratified_by_environment_and_length(
    dataset: Dataset,
    limit: int | None,
) -> Dataset:
    """Select a deterministic environment-proportional length-stratified sample."""
    if limit is None:
        return dataset
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("data.stratified_train_samples must be a positive integer.")
    limit = min(limit, len(dataset))
    groups: dict[str, list[int]] = {}
    for index in range(len(dataset)):
        source = dataset[index].get("source") or {}
        environment = str(source.get("environment", "unknown"))
        groups.setdefault(environment, []).append(index)

    exact_quotas = {
        environment: limit * len(indices) / len(dataset)
        for environment, indices in groups.items()
    }
    quotas = {
        environment: math.floor(exact_quota)
        for environment, exact_quota in exact_quotas.items()
    }
    remaining = limit - sum(quotas.values())
    for environment in sorted(
        groups,
        key=lambda name: (-(exact_quotas[name] - quotas[name]), name),
    )[:remaining]:
        quotas[environment] += 1

    selected: list[int] = []
    for environment in sorted(groups):
        indices = sorted(
            groups[environment],
            key=lambda index: (int(dataset[index]["_token_length"]), index),
        )
        quota = quotas[environment]
        if quota == 0:
            continue
        # Midpoints of equal-width bins cover the complete length distribution
        # without favoring either tail.
        positions = [
            min(
                len(indices) - 1,
                math.floor((i + 0.5) * len(indices) / quota),
            )
            for i in range(quota)
        ]
        selected.extend(indices[position] for position in positions)
    return dataset.select(selected)


def _benchmark_sample_manifest(dataset: Dataset) -> JsonObject:
    """Return a portable identity for the exact ordered benchmark sample."""
    records = [
        {
            "id": str(dataset[index]["id"]),
            "token_length": int(dataset[index]["_token_length"]),
            "supervised_tokens": int(dataset[index]["_supervised_tokens"]),
        }
        for index in range(len(dataset))
    ]
    ids = [record["id"] for record in records]
    return {
        "records": len(records),
        "ordered_record_ids": ids,
        "ordered_record_ids_sha256": sha256_text(
            json.dumps(ids, ensure_ascii=False, separators=(",", ":"))
        ),
        "ordered_records_sha256": sha256_text(
            json.dumps(records, ensure_ascii=False, separators=(",", ":"))
        ),
    }


def _load_prepared_split(
    path: Path,
    *,
    include_reasoning: bool,
    sample_limit: int | None,
    num_proc: int,
) -> Dataset:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared split does not exist: {path}")
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid prepared JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Prepared JSON in {path} at line {line_number} must be an object."
                )
            records.append(record)
    if not records:
        raise ValueError(f"Prepared split is empty: {path}")

    # Dynamic canonical fields stay as JSON instead of Arrow structs. The
    # streaming JSON loader fixes nested structs from its first chunk, making
    # heterogeneous source metadata depend on record ordering; inferred structs
    # also inject null message keys and change prepared chat tokenization.
    dataset = Dataset.from_list(records, features=_PREPARED_SPLIT_FEATURES)
    dataset = _limit_dataset(dataset, sample_limit)
    workers = min(max(1, num_proc), len(dataset))
    return dataset.map(
        _configure_example,
        fn_kwargs={"include_reasoning": include_reasoning},
        remove_columns=["messages"],
        num_proc=workers,
        desc=f"Configuring {'thinking' if include_reasoning else 'non-thinking'} messages",
    )


def _preflight_tokenization(
    dataset: Dataset,
    *,
    tokenizer: Any,
    training_template: str,
    num_proc: int,
) -> tuple[Dataset, JsonObject]:
    workers = min(max(1, num_proc), len(dataset))
    with_stats = dataset.map(
        _tokenization_stats,
        fn_kwargs={
            "tokenizer": tokenizer,
            "training_template": training_template,
        },
        num_proc=workers,
        desc="Validating assistant masks and lengths",
    )
    return with_stats, _summarize_tokenization(with_stats)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_prepared_file(
    manifest: Mapping[str, Any],
    *,
    filename: str,
    path: Path,
) -> None:
    prepared_files = manifest.get("prepared_files")
    if not isinstance(prepared_files, Mapping):
        raise ValueError("Prepared-data manifest has no prepared_files object.")
    metadata = prepared_files.get(filename)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Prepared-data manifest has no metadata for {filename}.")

    expected_bytes = metadata.get("bytes")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
        raise ValueError(
            f"Prepared-data manifest has invalid byte size for {filename}."
        )
    actual_bytes = path.stat().st_size
    if expected_bytes != actual_bytes:
        raise ValueError(
            f"Prepared-data file {path} has {actual_bytes} bytes, but the manifest "
            f"expects {expected_bytes}."
        )

    expected_sha256 = metadata.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"Prepared-data manifest has invalid SHA-256 for {filename}.")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 != actual_sha256:
        raise ValueError(
            f"Prepared-data file {path} has SHA-256 {actual_sha256}, but the manifest "
            f"expects {expected_sha256}."
        )


def _validate_manifest(
    manifest_path: Path,
    train_path: Path,
    validation_path: Path,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    limited: bool,
) -> JsonObject:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared-data manifest does not exist: {manifest_path}"
        )
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError(f"Prepared-data manifest {manifest_path} is not an object.")
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"Prepared-data manifest format_version must be {MANIFEST_FORMAT_VERSION}."
        )
    if manifest.get("canonical_schema_version") != CANONICAL_SCHEMA_VERSION:
        raise ValueError(
            "Prepared-data manifest has an unsupported canonical_schema_version."
        )
    identity = manifest.get("dataset")
    if not isinstance(identity, Mapping):
        raise ValueError("Prepared-data manifest has no dataset identity object.")
    for field in ("id", "version"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise ValueError(f"Prepared-data manifest has no valid dataset.{field}.")
    fingerprint = identity.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Prepared-data manifest has no valid dataset fingerprint.")
    actual_fingerprint = compute_dataset_fingerprint(manifest)
    if fingerprint != actual_fingerprint:
        raise ValueError(
            f"Prepared-data manifest fingerprint is {fingerprint}, but its contents "
            f"resolve to {actual_fingerprint}."
        )
    _validate_prepared_file(manifest, filename="train.jsonl", path=train_path)
    _validate_prepared_file(
        manifest,
        filename="validation.jsonl",
        path=validation_path,
    )
    if not limited:
        expected = manifest.get("records", {})
        if expected.get("train") != len(train_dataset):
            raise ValueError(
                f"Manifest expects {expected.get('train')} train records, loaded "
                f"{len(train_dataset)}."
            )
        if expected.get("validation") != len(validation_dataset):
            raise ValueError(
                f"Manifest expects {expected.get('validation')} validation records, loaded "
                f"{len(validation_dataset)}."
            )
    for split, dataset in (
        ("train", train_dataset),
        ("validation", validation_dataset),
    ):
        if "schema_version" not in dataset.column_names:
            raise ValueError(f"Prepared {split} records have no schema_version column.")
        invalid_versions = {
            value
            for value in dataset["schema_version"]
            if value != CANONICAL_SCHEMA_VERSION
        }
        if invalid_versions:
            raise ValueError(
                f"Prepared {split} records contain unsupported schema versions: "
                + ", ".join(map(str, sorted(invalid_versions, key=str)))
            )
    return manifest


def _validate_dataset_system_prompt_profile(
    manifest: Mapping[str, Any],
    *,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    expected_profile: Any,
) -> JsonObject | None:
    """Fail closed when a training config expects a particular prompt profile."""

    if expected_profile is None:
        return None
    if expected_profile not in DECOMPOSER_PROMPT_PROFILES:
        expected = ", ".join(DECOMPOSER_PROMPT_PROFILES)
        raise ValueError(
            "data.expected_system_prompt_profile must be one of: " + expected
        )
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("Prepared-data manifest has no policy object.")
    actual_profile = policy.get("system_prompt_profile")
    if actual_profile != expected_profile:
        raise ValueError(
            "Prepared-data prompt profile does not match "
            "data.expected_system_prompt_profile: "
            f"{actual_profile!r} != {expected_profile!r}."
        )
    expected_prompt = resolve_decomposer_system_prompt(expected_profile)
    expected_sha256 = sha256_text(expected_prompt)
    if policy.get("system_prompt_sha256") != expected_sha256:
        raise ValueError(
            "Prepared-data system prompt hash does not match the selected profile."
        )
    for split, dataset in (
        ("train", train_dataset),
        ("validation", validation_dataset),
    ):
        for index, messages in enumerate(dataset["messages"]):
            if (
                not messages
                or messages[0].get("role") != "system"
                or messages[0].get("content") != expected_prompt
            ):
                raise ValueError(
                    f"Prepared {split} record {index} does not start with the "
                    f"{expected_profile!r} Decomposer system prompt."
                )
    return {
        "profile": expected_profile,
        "sha256": expected_sha256,
    }


def _validate_prepared_tokenization(
    manifest: Mapping[str, Any],
    *,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    tokenizer: Any,
    training_template: str,
    model_name_or_path: str,
    model_revision: str,
    include_reasoning: bool,
    max_length: int | None,
    required: bool,
    profile: str = PREPARED_TOKENIZATION_PROFILE,
) -> JsonObject | None:
    if not isinstance(required, bool):
        raise ValueError("data.require_prepared_tokenization must be a boolean.")
    prepared = manifest.get("tokenization")
    if prepared is None:
        if required:
            raise ValueError(
                "The selected dataset has no mandatory prepared-tokenization metadata."
            )
        return None
    if not isinstance(prepared, Mapping):
        raise ValueError("Prepared-data manifest tokenization must be an object.")
    prepared = dict(prepared)
    if prepared.get("profile") != profile:
        raise ValueError("Prepared-data tokenization profile is unsupported.")
    if prepared.get("tokenizer") != model_name_or_path:
        raise ValueError(
            "Prepared-data tokenizer does not match model.name_or_path: "
            f"{prepared.get('tokenizer')!r} != {model_name_or_path!r}."
        )
    if prepared.get("requested_revision") != model_revision:
        raise ValueError(
            "Prepared-data tokenizer revision does not match model.revision."
        )
    if prepared.get("include_reasoning") is not include_reasoning:
        raise ValueError(
            "Prepared-data reasoning policy does not match data.include_reasoning."
        )
    prepared_max = prepared.get("max_tokens")
    if (
        isinstance(prepared_max, bool)
        or not isinstance(prepared_max, int)
        or prepared_max <= 0
    ):
        raise ValueError("Prepared-data tokenization has an invalid max_tokens.")
    if max_length is None:
        raise ValueError(
            "Prepared-tokenized datasets require training.max_length to be set."
        )
    if int(max_length) > prepared_max:
        raise ValueError(
            f"training.max_length={max_length} exceeds the prepared dataset ceiling "
            f"of {prepared_max} tokens."
        )
    expected_template_hash = hashlib.sha256(
        training_template.encode("utf-8")
    ).hexdigest()
    if prepared.get("training_template_sha256") != expected_template_hash:
        raise ValueError(
            "Prepared-data training template does not match the runtime template."
        )
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    runtime_revision = (
        init_kwargs.get("_commit_hash") if isinstance(init_kwargs, Mapping) else None
    )
    prepared_revision = prepared.get("resolved_revision")
    if prepared_revision != runtime_revision:
        raise ValueError(
            "Prepared-data resolved tokenizer revision does not match the runtime "
            "tokenizer revision."
        )

    for split, dataset in (
        ("train", train_dataset),
        ("validation", validation_dataset),
    ):
        for index, example in enumerate(dataset):
            attributes = example.get("attributes")
            metadata = (
                attributes.get(PREPARED_TOKENIZATION_ATTRIBUTE)
                if isinstance(attributes, Mapping)
                else None
            )
            if not isinstance(metadata, Mapping):
                raise ValueError(
                    f"Prepared {split} example {example.get('id', index)} has no "
                    "mandatory token metadata."
                )
            if metadata.get("profile") != profile:
                raise ValueError(
                    f"Prepared {split} example {example.get('id', index)} has an "
                    "unsupported tokenization profile."
                )
            if metadata.get("tokens") != int(example["_token_length"]):
                raise ValueError(
                    f"Prepared {split} example {example.get('id', index)} token count "
                    "does not match fresh tokenization."
                )
            if metadata.get("supervised_tokens") != int(example["_supervised_tokens"]):
                raise ValueError(
                    f"Prepared {split} example {example.get('id', index)} supervised "
                    "token count does not match fresh tokenization."
                )
    return prepared


def _apply_overlength_policy(
    dataset: Dataset,
    *,
    split: str,
    max_length: int | None,
    exclude_overlength: bool,
    error_on_truncation: bool,
) -> tuple[Dataset, list[JsonObject]]:
    if not isinstance(exclude_overlength, bool):
        raise ValueError("data.exclude_overlength must be a boolean.")
    if not isinstance(error_on_truncation, bool):
        raise ValueError("data.error_on_truncation must be a boolean.")
    if max_length is None:
        if exclude_overlength:
            raise ValueError(
                "data.exclude_overlength requires training.max_length to be set."
            )
        return dataset, []

    limit = int(max_length)
    excluded: list[JsonObject] = []
    retained_indices: list[int] = []
    for index, example in enumerate(dataset):
        token_length = int(example["_token_length"])
        if token_length <= limit:
            retained_indices.append(index)
            continue
        excluded.append(
            {
                "id": str(example.get("id", f"{split}:{index}")),
                "split": split,
                "token_length": token_length,
                "max_length": limit,
            }
        )

    if not excluded:
        return dataset, []
    if not exclude_overlength:
        if error_on_truncation:
            longest = max(item["token_length"] for item in excluded)
            raise ValueError(
                f"The longest prepared trace has {longest} tokens, exceeding "
                f"max_length={limit}. Increase max_length or explicitly enable "
                "data.exclude_overlength; traces are never silently truncated by "
                "the checked-in configs."
            )
        return dataset, []
    if not retained_indices:
        raise ValueError(
            f"Excluding traces longer than {limit} tokens emptied the {split} split."
        )
    return dataset.select(retained_indices), excluded


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_rank_zero() -> bool:
    return _rank() == 0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _resolve_train_batch_config(
    training_config: Mapping[str, Any],
    *,
    world_size: int,
) -> tuple[JsonObject, JsonObject]:
    """Resolve optimizer-step batch semantics before constructing SFTConfig."""
    resolved = dict(training_config)
    world_size = _positive_integer(world_size, field="run.expected_world_size")
    per_device_batch = _positive_integer(
        resolved.get("per_device_train_batch_size"),
        field="training.per_device_train_batch_size",
    )
    if "gradient_accumulation_steps" in resolved:
        raise ValueError(
            "training.gradient_accumulation_steps is derived from "
            "training.global_batch_size and must not be specified."
        )
    if "global_batch_size" not in resolved:
        raise ValueError("training.global_batch_size is required.")
    global_batch = _positive_integer(
        resolved.pop("global_batch_size"),
        field="training.global_batch_size",
    )
    micro_batch = world_size * per_device_batch
    if global_batch % micro_batch != 0:
        raise ValueError(
            f"training.global_batch_size={global_batch} must be divisible by "
            "run.expected_world_size * training.per_device_train_batch_size "
            f"= {micro_batch}."
        )
    accumulation = global_batch // micro_batch
    resolved["gradient_accumulation_steps"] = accumulation

    return resolved, {
        "global_batch_size": global_batch,
        "world_size": world_size,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": accumulation,
    }


def _generation_eos_ids(value: int | Sequence[int] | None) -> list[int]:
    if value is None:
        return []
    values = (
        [value] if isinstance(value, int) and not isinstance(value, bool) else value
    )
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("generation_config.eos_token_id must contain token IDs.")
    eos_ids: list[int] = []
    for token_id in values:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError("generation_config.eos_token_id must contain token IDs.")
        if token_id not in eos_ids:
            eos_ids.append(token_id)
    return eos_ids


def _configure_gemma4_generation(
    generation_config: GenerationConfig,
    *,
    tokenizer: Any,
) -> JsonObject:
    """Align tokenizer IDs and preserve every required Gemma-4 stop token."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or eos_token_id < 0
    ):
        raise ValueError("The Gemma-4 tokenizer must define a valid eos_token_id.")

    eos_ids = [eos_token_id]
    for token_id in _generation_eos_ids(generation_config.eos_token_id):
        if token_id not in eos_ids:
            eos_ids.append(token_id)

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    required_stop_ids: dict[str, int] = {}
    for token in _GEMMA4_REQUIRED_STOP_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        roundtrip = (
            tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(token_id, int) and not isinstance(token_id, bool)
            else None
        )
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id == unk_token_id
            or roundtrip != token
        ):
            raise ValueError(
                f"Required Gemma-4 stop token {token!r} is not an exact tokenizer token."
            )
        required_stop_ids[token] = token_id
        if token_id not in eos_ids:
            eos_ids.append(token_id)

    generation_config.bos_token_id = tokenizer.bos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = eos_ids
    return {
        "bos_token_id": generation_config.bos_token_id,
        "eos_token_id": eos_ids,
        "pad_token_id": generation_config.pad_token_id,
        "required_stop_token_ids": required_stop_ids,
    }


def _configure_qwen35_generation(
    generation_config: GenerationConfig,
    *,
    tokenizer: Any,
) -> JsonObject:
    """Align Qwen3.5 generation metadata with its tokenizer and end markers."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or eos_token_id < 0
    ):
        raise ValueError("The Qwen3.5 tokenizer must define a valid eos_token_id.")
    if (
        isinstance(pad_token_id, bool)
        or not isinstance(pad_token_id, int)
        or pad_token_id < 0
    ):
        raise ValueError("The Qwen3.5 tokenizer must define a valid pad_token_id.")
    eos_ids = [eos_token_id]
    for token_id in _generation_eos_ids(generation_config.eos_token_id):
        if token_id not in eos_ids:
            eos_ids.append(token_id)
    generation_config.bos_token_id = tokenizer.bos_token_id
    generation_config.pad_token_id = pad_token_id
    generation_config.eos_token_id = eos_ids
    return {
        "bos_token_id": generation_config.bos_token_id,
        "eos_token_id": eos_ids,
        "pad_token_id": pad_token_id,
        "required_stop_token_ids": {},
    }


def _load_generation_config(
    model_name_or_path: str,
    *,
    revision: str,
    model_config: Any,
) -> tuple[GenerationConfig, str]:
    try:
        return (
            GenerationConfig.from_pretrained(
                model_name_or_path,
                revision=revision,
            ),
            "pretrained",
        )
    except OSError:
        # Qwen3.5 base snapshots may omit generation_config.json. Transformers'
        # model-derived defaults still preserve checkpoint EOS metadata.
        return GenerationConfig.from_model_config(model_config), "model_config"


def _configure_generation(
    profile: str,
    generation_config: GenerationConfig,
    *,
    tokenizer: Any,
) -> JsonObject:
    if profile == "gemma4_sft_non_thinking":
        return _configure_gemma4_generation(generation_config, tokenizer=tokenizer)
    if profile == "qwen35_sft_non_thinking":
        return _configure_qwen35_generation(generation_config, tokenizer=tokenizer)
    raise ValueError(f"Unsupported prepared tokenization profile: {profile}")


def _save_final_configuration(
    final_dir: Path,
    *,
    model_config: Any,
    tokenizer: Any,
    generation_config: GenerationConfig,
) -> None:
    """Save inference metadata that is independent of the weight export path."""
    final_dir.mkdir(parents=True, exist_ok=True)
    model_config.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    generation_config.save_pretrained(final_dir)


def _has_existing_run_output(output_dir: Path) -> bool:
    """Ignore launcher-owned logs while protecting every training artifact."""
    return output_dir.exists() and any(
        path.name not in _LAUNCHER_LOG_FILENAMES for path in output_dir.iterdir()
    )


def _configure_sdpa_backends(run_config: Mapping[str, Any]) -> JsonObject:
    """Apply per-process SDPA policy and return the resolved backend state."""
    disable_cudnn = run_config.get("disable_cudnn_sdpa", False)
    if not isinstance(disable_cudnn, bool):
        raise ValueError("run.disable_cudnn_sdpa must be a boolean.")

    cuda_backends = torch.backends.cuda
    if disable_cudnn:
        cuda_backends.enable_cudnn_sdp(False)

    return {
        "cudnn": bool(cuda_backends.cudnn_sdp_enabled()),
        "flash": bool(cuda_backends.flash_sdp_enabled()),
        "memory_efficient": bool(cuda_backends.mem_efficient_sdp_enabled()),
        "math": bool(cuda_backends.math_sdp_enabled()),
    }


def _build_early_stopping_callback(
    run_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> EarlyStoppingCallback | None:
    early_stopping = run_config.get("early_stopping")
    if early_stopping is None:
        return None
    if not isinstance(early_stopping, Mapping):
        raise ValueError("run.early_stopping must be an object when specified.")

    unknown = set(early_stopping) - {"patience", "threshold"}
    if unknown:
        raise ValueError(
            "Unknown run.early_stopping fields: " + ", ".join(sorted(unknown))
        )

    patience = early_stopping.get("patience")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
        raise ValueError("run.early_stopping.patience must be a positive integer.")
    threshold = early_stopping.get("threshold", 0.0)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) < 0.0
    ):
        raise ValueError(
            "run.early_stopping.threshold must be a finite non-negative number."
        )

    eval_strategy = str(training_config.get("eval_strategy", "no")).lower()
    save_strategy = str(training_config.get("save_strategy", "no")).lower()
    if eval_strategy == "no" or save_strategy != eval_strategy:
        raise ValueError(
            "Early stopping requires matching, enabled training.eval_strategy and "
            "training.save_strategy values."
        )
    if not training_config.get("load_best_model_at_end", False):
        raise ValueError(
            "Early stopping requires training.load_best_model_at_end=true."
        )
    if training_config.get("metric_for_best_model") != "eval_loss":
        raise ValueError(
            "Early stopping requires training.metric_for_best_model=eval_loss."
        )
    if training_config.get("greater_is_better") is not False:
        raise ValueError(
            "Early stopping on eval_loss requires greater_is_better=false."
        )

    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=float(threshold),
    )


def _summarize_trainer_state(
    trainer: SFTTrainer,
    *,
    early_stopping_enabled: bool,
) -> JsonObject:
    state = trainer.state
    global_step = int(state.global_step)
    max_steps = int(state.max_steps)
    return {
        "global_step": global_step,
        "max_steps": max_steps,
        "completed_epochs": None if state.epoch is None else float(state.epoch),
        "best_metric": (
            None if state.best_metric is None else float(state.best_metric)
        ),
        "best_model_checkpoint": (
            None
            if state.best_model_checkpoint is None
            else str(state.best_model_checkpoint)
        ),
        "early_stopped": early_stopping_enabled and global_step < max_steps,
    }


def _build_trainer_callbacks(
    clearml_config: Mapping[str, Any],
    early_stopping_callback: EarlyStoppingCallback | None,
) -> list[Any]:
    callbacks: list[Any] = []
    if early_stopping_callback is not None:
        callbacks.append(early_stopping_callback)
    if clearml_config.get("enabled"):
        # Instantiate this on every rank: FSDP weight norms require all ranks to
        # enter the sharded reduction even though only rank zero reports them.
        callbacks.append(
            SeparatePlotsClearMLCallback(
                weight_norm_interval_steps=clearml_config["weight_norm_interval_steps"]
            )
        )
    return callbacks


def _initialize_clearml(config: Mapping[str, Any], resolved_config: JsonObject):
    if not config.get("enabled", False):
        return None
    if not _is_rank_zero():
        return None
    if not (
        os.environ.get("CLEARML_API_HOST")
        or os.environ.get("CLEARML_CONFIG_FILE")
        or (Path.home() / "clearml.conf").is_file()
    ):
        raise RuntimeError(
            "ClearML is enabled, but no self-hosted configuration was found. Set "
            "CLEARML_API_HOST plus credentials, set CLEARML_CONFIG_FILE, or create "
            "~/clearml.conf."
        )

    os.environ.setdefault("CLEARML_NO_DEFAULT_SERVER", "1")
    tags = config.get("tags", [])
    if (
        not isinstance(tags, Sequence)
        or isinstance(tags, (str, bytes))
        or not all(isinstance(tag, str) and tag for tag in tags)
    ):
        raise ValueError("clearml.tags must be a list of non-empty strings.")

    os.environ["CLEARML_PROJECT"] = str(config.get("project", "decomposer"))
    os.environ["CLEARML_TASK"] = str(config.get("task", "Decomposer SFT"))
    os.environ["CLEARML_LOG_MODEL"] = (
        "TRUE" if config.get("log_model", False) else "FALSE"
    )

    from clearml import Task

    task = Task.init(
        project_name=os.environ["CLEARML_PROJECT"],
        task_name=os.environ["CLEARML_TASK"],
        tags=list(tags),
        reuse_last_task_id=False,
        auto_connect_frameworks={"tensorboard": False, "pytorch": False},
        output_uri=config.get("output_uri") or False,
    )
    task.connect(resolved_config, name="Decomposer SFT")
    return task


def _resolve_config(config: JsonObject, args: argparse.Namespace) -> JsonObject:
    config = json.loads(json.dumps(config))
    model = _nested_mapping(config, "model")
    data = _nested_mapping(config, "data")
    training = _nested_mapping(config, "training")
    clearml = _nested_mapping(config, "clearml")
    run = _nested_mapping(config, "run")

    if args.include_reasoning is not None:
        data["include_reasoning"] = args.include_reasoning
    if args.max_length is not None:
        training["max_length"] = args.max_length
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
    if args.use_liger_kernel is not None:
        training["use_liger_kernel"] = args.use_liger_kernel
    if args.output_dir is not None:
        training["output_dir"] = args.output_dir
    if args.clearml is not None:
        clearml["enabled"] = args.clearml
    if args.resume_from_checkpoint is not None:
        run["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.max_train_samples is not None:
        data["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        data["max_eval_samples"] = args.max_eval_samples
    if args.train_file is not None:
        data["train_file"] = args.train_file
    if args.validation_file is not None:
        data["validation_file"] = args.validation_file
    if args.manifest_file is not None:
        data["manifest_file"] = args.manifest_file
    if args.longest_train_samples is not None:
        data["longest_train_samples"] = args.longest_train_samples
    if args.stratified_train_samples is not None:
        data["stratified_train_samples"] = args.stratified_train_samples
    if args.per_device_train_batch_size is not None:
        training["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.global_batch_size is not None:
        training["global_batch_size"] = args.global_batch_size
    if args.group_by_length is not None:
        training["train_sampling_strategy"] = (
            "group_by_length" if args.group_by_length else "random"
        )
        if args.group_by_length:
            training["length_column_name"] = "_token_length"
    if args.attn_implementation is not None:
        model["attn_implementation"] = args.attn_implementation
    if args.benchmark:
        if (
            data.get("longest_train_samples") is not None
            and data.get("stratified_train_samples") is not None
        ):
            raise ValueError(
                "Benchmark selection accepts either longest or stratified samples, not both."
            )
        run["benchmark"] = True
        run["early_stopping"] = None
        training["num_train_epochs"] = 1
        training["eval_strategy"] = "no"
        training["save_strategy"] = "no"
        training["load_best_model_at_end"] = False
        clearml["enabled"] = False

    clearml["weight_norm_interval_steps"] = validate_weight_norm_interval(
        clearml.get("weight_norm_interval_steps", 10)
    )

    training["loss_type"] = liger_compatible_loss_type(
        bool(training.get("use_liger_kernel", False)),
        training.get("loss_type"),
    )

    config["model"] = model
    config["data"] = data
    config["training"] = training
    config["clearml"] = clearml
    config["run"] = run
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-parameter SFT on Decomposer traces."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--include-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--use-liger-kernel",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--clearml", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--train-file")
    parser.add_argument("--validation-file")
    parser.add_argument("--manifest-file")
    parser.add_argument("--longest-train-samples", type=int)
    parser.add_argument("--stratified-train-samples", type=int)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument(
        "--group-by-length",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Group similarly sized examples in the Trainer sampler.",
    )
    parser.add_argument("--attn-implementation")
    parser.add_argument("--benchmark", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    raw_config = _load_yaml(args.config.resolve())
    config = _resolve_config(raw_config, args)
    model_config = _nested_mapping(config, "model")
    data_config = _nested_mapping(config, "data")
    training_config = _nested_mapping(config, "training")
    clearml_config = _nested_mapping(config, "clearml")
    run_config = _nested_mapping(config, "run")
    expected_world_size = _positive_integer(
        run_config.get("expected_world_size", 2),
        field="run.expected_world_size",
    )
    training_config, batch_runtime = _resolve_train_batch_config(
        training_config,
        world_size=expected_world_size,
    )
    config["training"] = training_config
    sdpa_backends = _configure_sdpa_backends(run_config)
    if _is_rank_zero():
        print(f"Resolved SDPA backends: {json.dumps(sdpa_backends, sort_keys=True)}")
        print(f"Resolved train batch: {json.dumps(batch_runtime, sort_keys=True)}")
    early_stopping_callback = _build_early_stopping_callback(
        run_config,
        training_config,
    )

    if _world_size() != expected_world_size:
        raise RuntimeError(
            f"This config expects {expected_world_size} processes, but WORLD_SIZE is "
            f"{_world_size()}. Launch it with torchrun --nproc-per-node={expected_world_size}."
        )

    required_runtime = _validate_required_runtime_profile(run_config, model_config)
    if _is_rank_zero() and required_runtime is not None:
        print(
            "Validated required runtime profile: "
            + json.dumps(
                {
                    "profile": required_runtime["profile"],
                    "packages": required_runtime["packages"],
                    "cuda_device": required_runtime["cuda_device"],
                    "cuda_capability": required_runtime["cuda_capability"],
                },
                sort_keys=True,
            )
        )

    model_name_or_path = str(model_config["name_or_path"])
    include_reasoning = bool(data_config.get("include_reasoning", False))
    model_revision = str(model_config.get("revision", "main"))
    checkpoint_config = AutoConfig.from_pretrained(
        model_name_or_path,
        revision=model_revision,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    tokenization_profile = tokenization_profile_for_model_config(checkpoint_config)
    validate_reasoning_policy(
        tokenization_profile,
        include_reasoning=include_reasoning,
    )
    num_proc = int(data_config.get("num_proc", 8))
    max_train_samples = data_config.get("max_train_samples")
    max_eval_samples = data_config.get("max_eval_samples")

    train_path = Path(data_config["train_file"]).resolve()
    validation_path = Path(data_config["validation_file"]).resolve()
    manifest_path = Path(data_config["manifest_file"]).resolve()
    train_dataset = _load_prepared_split(
        train_path,
        include_reasoning=include_reasoning,
        sample_limit=max_train_samples,
        num_proc=num_proc,
    )
    validation_dataset = _load_prepared_split(
        validation_path,
        include_reasoning=include_reasoning,
        sample_limit=max_eval_samples,
        num_proc=num_proc,
    )
    limited = max_train_samples is not None or max_eval_samples is not None
    manifest = _validate_manifest(
        manifest_path,
        train_path,
        validation_path,
        train_dataset,
        validation_dataset,
        limited=limited,
    )
    system_prompt_runtime = _validate_dataset_system_prompt_profile(
        manifest,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        expected_profile=data_config.get("expected_system_prompt_profile"),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=model_config.get("revision", "main"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    generation_config, generation_config_source = _load_generation_config(
        model_name_or_path,
        revision=model_revision,
        model_config=checkpoint_config,
    )
    generation_runtime = _configure_generation(
        tokenization_profile,
        generation_config,
        tokenizer=tokenizer,
    )
    generation_runtime["source"] = generation_config_source
    tokenizer.padding_side = "right"
    canonical_template = tokenizer.chat_template
    training_template = build_training_template(
        tokenization_profile,
        canonical_template,
    )
    tokenizer.chat_template = training_template

    train_dataset, raw_train_token_stats = _preflight_tokenization(
        train_dataset,
        tokenizer=tokenizer,
        training_template=training_template,
        num_proc=num_proc,
    )
    validation_dataset, raw_validation_token_stats = _preflight_tokenization(
        validation_dataset,
        tokenizer=tokenizer,
        training_template=training_template,
        num_proc=num_proc,
    )
    max_length = training_config.get("max_length")
    prepared_tokenization = _validate_prepared_tokenization(
        manifest,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        tokenizer=tokenizer,
        training_template=training_template,
        model_name_or_path=model_name_or_path,
        model_revision=str(model_config.get("revision", "main")),
        include_reasoning=include_reasoning,
        max_length=max_length,
        required=data_config.get("require_prepared_tokenization", False),
        profile=tokenization_profile,
    )
    exclude_overlength = data_config.get("exclude_overlength", False)
    error_on_truncation = data_config.get("error_on_truncation", True)
    train_dataset, train_overlength_exclusions = _apply_overlength_policy(
        train_dataset,
        split="train",
        max_length=max_length,
        exclude_overlength=exclude_overlength,
        error_on_truncation=error_on_truncation,
    )
    validation_dataset, validation_overlength_exclusions = _apply_overlength_policy(
        validation_dataset,
        split="validation",
        max_length=max_length,
        exclude_overlength=exclude_overlength,
        error_on_truncation=error_on_truncation,
    )
    overlength_exclusions = [
        *train_overlength_exclusions,
        *validation_overlength_exclusions,
    ]
    longest_train_samples = data_config.get("longest_train_samples")
    train_dataset = _select_longest_by_token_length(
        train_dataset,
        longest_train_samples,
    )
    stratified_train_samples = data_config.get("stratified_train_samples")
    train_dataset = _select_stratified_by_environment_and_length(
        train_dataset,
        stratified_train_samples,
    )
    train_token_stats = (
        _summarize_tokenization(train_dataset)
        if train_overlength_exclusions
        or longest_train_samples is not None
        or stratified_train_samples is not None
        else raw_train_token_stats
    )
    validation_token_stats = (
        _summarize_tokenization(validation_dataset)
        if validation_overlength_exclusions
        else raw_validation_token_stats
    )
    if _is_rank_zero() and overlength_exclusions:
        print(
            "Excluded overlength prepared traces: "
            + json.dumps(overlength_exclusions, sort_keys=True)
        )

    training_config["assistant_only_loss"] = True
    training_config["packing"] = False
    training_config["dataset_num_proc"] = num_proc
    # A custom ClearML callback preserves the integration lifecycle while giving
    # every scalar its own plot instead of grouping all train/eval series.
    training_config["report_to"] = []
    model_init_kwargs = dict(training_config.get("model_init_kwargs") or {})
    model_init_kwargs.setdefault("dtype", model_config.get("dtype", "bfloat16"))
    model_init_kwargs.setdefault(
        "attn_implementation", model_config.get("attn_implementation", "sdpa")
    )
    model_init_kwargs.setdefault("revision", model_config.get("revision", "main"))
    training_config["model_init_kwargs"] = model_init_kwargs

    output_dir = Path(training_config["output_dir"]).resolve()
    resume_from_checkpoint = run_config.get("resume_from_checkpoint")
    if (
        _is_rank_zero()
        and _has_existing_run_output(output_dir)
        and not resume_from_checkpoint
    ):
        if not run_config.get("overwrite_output_dir", False):
            raise FileExistsError(
                f"Output directory {output_dir} is non-empty. Set a resume checkpoint or "
                "explicitly enable run.overwrite_output_dir."
            )

    liger_runtime = configure_liger_for_model(
        model_name_or_path,
        revision=str(model_config.get("revision", "main")),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        enabled=bool(training_config.get("use_liger_kernel", False)),
        kernel_config=training_config.get("liger_kernel_config"),
    )

    resolved = {
        **config,
        "model": model_config,
        "data": data_config,
        "training": training_config,
        "clearml": clearml_config,
        "run": run_config,
        "runtime": {
            "world_size": _world_size(),
            "git_revision": _git_revision(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "train_batch": batch_runtime,
            "generation_config": generation_runtime,
            "sdpa_backends": sdpa_backends,
            "required_runtime_profile": required_runtime,
            "liger_kernel": liger_runtime,
            "include_reasoning": include_reasoning,
            "system_prompt": system_prompt_runtime,
            "tokenization_profile": tokenization_profile,
            "prepared_tokenization": prepared_tokenization,
            "raw_train_token_stats": raw_train_token_stats,
            "raw_validation_token_stats": raw_validation_token_stats,
            "train_token_stats": train_token_stats,
            "validation_token_stats": validation_token_stats,
            "overlength_exclusions": overlength_exclusions,
            "effective_records": {
                "train": len(train_dataset),
                "validation": len(validation_dataset),
            },
        },
    }

    clearml_task = _initialize_clearml(clearml_config, resolved)
    if clearml_task is not None:
        clearml_task.upload_artifact(
            "prepared-data-manifest", artifact_object=str(manifest_path)
        )

    if _is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "resolved_config.json", resolved)
        shutil.copy2(manifest_path, output_dir / "data_manifest.json")
        (output_dir / "training_chat_template.jinja").write_text(
            training_template,
            encoding="utf-8",
        )

    sft_config = SFTConfig(**training_config)
    callbacks = _build_trainer_callbacks(
        clearml_config,
        early_stopping_callback,
    )
    trainer = SFTTrainer(
        model=model_name_or_path,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=callbacks or None,
    )
    trainer.model.generation_config = generation_config

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        benchmark_started = time.perf_counter()
        train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        benchmark_elapsed = time.perf_counter() - benchmark_started
        training_state = _summarize_trainer_state(
            trainer,
            early_stopping_enabled=early_stopping_callback is not None,
        )
        local_runtime = {
            "rank": _rank(),
            "elapsed_seconds": benchmark_elapsed,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            ),
            "allocated_bytes_after_train": (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            ),
            "reserved_bytes_after_train": (
                torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            ),
        }
        rank_runtimes: list[JsonObject | None] = [None] * _world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_gather_object(rank_runtimes, local_runtime)
        else:
            rank_runtimes = [local_runtime]

        if run_config.get("benchmark", False):
            if _is_rank_zero():
                summary = {
                    "benchmark": True,
                    "train_metrics": train_output.metrics,
                    "training_state": training_state,
                    "rank_runtimes": rank_runtimes,
                    "train_token_stats": train_token_stats,
                    "train_batch": batch_runtime,
                    "train_sampling_strategy": training_config.get(
                        "train_sampling_strategy", "random"
                    ),
                    "sample": _benchmark_sample_manifest(train_dataset),
                    "attention_implementation": model_config.get(
                        "attn_implementation", "sdpa"
                    ),
                    "attention_backend_runtime": _attention_backend_runtime(
                        str(model_config.get("attn_implementation", "sdpa")),
                        resolved_implementation=getattr(
                            trainer.model.config,
                            "_attn_implementation",
                            None,
                        ),
                    ),
                    "linear_attention_runtime": _qwen35_linear_attention_runtime(),
                    "effective_data": {
                        "train": len(train_dataset),
                        "validation": len(validation_dataset),
                    },
                }
                _write_json(output_dir / "benchmark_summary.json", summary)
            return

        tokenizer.chat_template = canonical_template
        final_dir = output_dir / "final"
        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        uses_sharded_state = fsdp_plugin is not None and "SHARDED_STATE_DICT" in str(
            fsdp_plugin.state_dict_type
        )
        if uses_sharded_state:
            save_fsdp_model(
                fsdp_plugin,
                trainer.accelerator,
                trainer.model,
                str(final_dir),
            )
        else:
            trainer.save_model(str(final_dir))

        if _is_rank_zero():
            _save_final_configuration(
                final_dir,
                model_config=trainer.model.config,
                tokenizer=tokenizer,
                generation_config=generation_config,
            )

        if uses_sharded_state:
            accelerator = trainer.accelerator
            accelerator.wait_for_everyone()
            accelerator.free_memory(trainer.model, trainer.optimizer)
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            # Accelerate performs the write only on its main process, but this
            # helper ends with a distributed barrier. Every rank must enter it.
            merge_fsdp_weights(
                final_dir / "pytorch_model_fsdp_0",
                final_dir,
                safe_serialization=True,
                remove_checkpoint_dir=True,
            )

        if _is_rank_zero():
            summary = {
                "train_metrics": train_output.metrics,
                "training_state": training_state,
                "train_token_stats": train_token_stats,
                "validation_token_stats": validation_token_stats,
                "overlength_exclusions": overlength_exclusions,
                "train_batch": batch_runtime,
                "generation_config": generation_runtime,
                "final_model_dir": str(final_dir),
                "prepared_data": manifest.get("records"),
                "effective_data": {
                    "train": len(train_dataset),
                    "validation": len(validation_dataset),
                },
            }
            _write_json(output_dir / "training_summary.json", summary)
            if clearml_task is not None:
                clearml_task.upload_artifact(
                    "training-summary", artifact_object=summary
                )
    finally:
        if clearml_task is not None:
            clearml_task.close()


if __name__ == "__main__":
    main()
