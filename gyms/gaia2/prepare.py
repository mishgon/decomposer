"""Prepare pinned Gaia2 execution data, runtime, models, and manifests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gyms.gaia2.dataset import (  # noqa: E402
    load_huggingface_rows,
    sha256_file,
    validate_materialized_dataset,
    validate_materialized_filesystem,
    write_filesystem_manifest,
    write_materialized_dataset,
)
from gyms.gaia2.experiments import (  # noqa: E402
    DATASET_REVISION,
    DEFAULT_GAIA2_REPO,
    DOMAIN,
    FILESYSTEM_DATASET_ID,
    FILESYSTEM_DATASET_REVISION,
    GAIA2_REVISION,
    GAIA2_STAGING_ROOT,
    HF_HOME,
    PARTITIONS,
    PROJECT_VENV,
    SPLIT,
    SPLIT_MANIFEST_NAME,
    UV_BIN,
    UV_CACHE,
    DecomposerExperiment,
    Experiment,
    SimpleExperiment,
    collect_experiments,
    dataset_manifest,
    dataset_revision_root,
    filesystem_manifest,
    filesystem_revision_root,
    gaia2_venv,
    preparation_manifest,
)
from gyms.gaia2.partition import (  # noqa: E402
    SPLIT_MANIFEST_RELPATH,
    SPLIT_MANIFEST_SHA256,
    materialize_partition_views,
)
from gyms.gaia2.staging import resolve_revision, stage_revision  # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _uv_executable() -> Path:
    if UV_BIN.is_file() and os.access(UV_BIN, os.X_OK):
        return UV_BIN
    discovered = shutil.which("uv")
    if discovered:
        return Path(discovered)
    raise FileNotFoundError("uv is required to prepare the Gaia2 runtime")


def prepare_gaia2_runtime(staged_gaia2: Path, *, create: bool) -> dict[str, Any]:
    uv = _uv_executable()
    venv = gaia2_venv(staged_gaia2)
    benchmark = venv / "bin" / "are-benchmark"
    if not benchmark.is_file() and create:
        venv.parent.mkdir(parents=True, exist_ok=True)
        UV_CACHE.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(venv),
            "UV_CACHE_DIR": str(UV_CACHE),
            "UV_LINK_MODE": "copy",
        }
        subprocess.run(
            [
                str(uv),
                "sync",
                "--project",
                str(staged_gaia2),
                "--frozen",
                "--python",
                "3.12",
            ],
            check=True,
            env=env,
        )
    if not benchmark.is_file():
        raise FileNotFoundError(
            f"Prepared Gaia2 executable is missing: {benchmark}; rerun without --skip-runtime"
        )
    lock = staged_gaia2 / "uv.lock"
    return {
        "path": str(venv),
        "python": str(venv / "bin" / "python"),
        "are_benchmark": str(benchmark),
        "uv_lock_sha256": sha256_file(lock),
        "uv_bin": str(uv),
        "uv_sha256": sha256_file(uv),
    }


def validate_checkpoint(
    checkpoint: Path, *, full_hashes: bool = False
) -> dict[str, Any]:
    required_metadata = {"config.json", "tokenizer_config.json"}
    missing = [name for name in required_metadata if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{checkpoint}: missing {', '.join(sorted(missing))}")
    index_path = checkpoint / "model.safetensors.index.json"
    monolithic_path = checkpoint / "model.safetensors"
    required_hashes = set(required_metadata)
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise ValueError(f"{index_path}: weight_map has no shards")
        required_hashes.add(index_path.name)
    elif monolithic_path.is_file():
        shards = [monolithic_path.name]
    else:
        raise FileNotFoundError(
            f"{checkpoint}: missing model.safetensors or model.safetensors.index.json"
        )
    for shard in shards:
        path = checkpoint / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{checkpoint}: missing or empty shard {shard}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in checkpoint.iterdir() if item.is_file()):
        entry: dict[str, Any] = {"size": path.stat().st_size}
        if full_hashes or path.name in required_hashes:
            entry["sha256"] = sha256_file(path)
        files[path.name] = entry
    return {
        "path": str(checkpoint),
        "safetensors_shards": shards,
        "files": files,
    }


def experiment_models(experiment: Experiment, *, full_hashes: bool) -> dict[str, Any]:
    if isinstance(experiment, SimpleExperiment):
        if experiment.requires_openrouter:
            return {
                "policy": {
                    "backend": experiment.backend,
                    "model": experiment.served_name,
                }
            }
        if experiment.checkpoint is None:
            raise ValueError("Local simple agent requires checkpoint")
        return {
            "policy": validate_checkpoint(
                experiment.checkpoint, full_hashes=full_hashes
            )
        }
    models = {
        "worker": validate_checkpoint(
            experiment.worker_checkpoint, full_hashes=full_hashes
        )
    }
    if experiment.requires_local_manager:
        if experiment.manager_checkpoint is None:
            raise ValueError("Local manager requires manager_checkpoint")
        models["manager"] = validate_checkpoint(
            experiment.manager_checkpoint, full_hashes=full_hashes
        )
    else:
        models["manager"] = {
            "backend": experiment.manager_backend,
            "model": experiment.manager_served_name,
        }
    return models


def _prepare_dataset(*, reuse_source: bool, gaia2_revision: str) -> dict[str, Any]:
    revision_root = dataset_revision_root()
    if revision_root.exists():
        return validate_materialized_dataset(revision_root)
    if reuse_source:
        raise FileNotFoundError(f"Prepared Gaia2 data is missing: {dataset_manifest()}")
    rows = load_huggingface_rows(cache_dir=HF_HOME / "datasets")
    return write_materialized_dataset(
        rows,
        revision_root,
        gaia2_revision=gaia2_revision,
    )


def _prepare_filesystem(*, reuse_source: bool) -> dict[str, Any]:
    revision_root = filesystem_revision_root()
    if revision_root.exists():
        return validate_materialized_filesystem(revision_root)
    if reuse_source:
        raise FileNotFoundError(
            f"Prepared Gaia2 filesystem is missing: {filesystem_manifest()}"
        )

    from huggingface_hub import snapshot_download

    revision_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = revision_root.with_name(f".{revision_root.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"Incomplete Gaia2 filesystem download exists: {temporary}")
    temporary.mkdir()
    try:
        snapshot_download(
            repo_id=FILESYSTEM_DATASET_ID,
            repo_type="dataset",
            revision=FILESYSTEM_DATASET_REVISION,
            local_dir=temporary,
            allow_patterns=("demo_filesystem/**",),
        )
        write_filesystem_manifest(temporary)
        os.rename(temporary, revision_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_materialized_filesystem(revision_root)


def prepare_eval(args: argparse.Namespace) -> int:
    if args.purpose == "trace-generation" and args.partition != "train":
        raise ValueError(
            "Gaia2 trace generation preparation is restricted to the train partition"
        )
    experiments = collect_experiments(
        tuple(args.experiment or ()), tuple(args.filter or ())
    )
    if not experiments:
        raise ValueError("Experiment selectors matched no registered experiments")

    gaia2_source = args.gaia_repo.expanduser().resolve()
    commit = resolve_revision(gaia2_source, GAIA2_REVISION)
    if commit != GAIA2_REVISION:
        raise ValueError(
            f"Gaia2 revision resolved to {commit}, expected {GAIA2_REVISION}"
        )
    staged_gaia2 = stage_revision(gaia2_source, commit, GAIA2_STAGING_ROOT / commit)
    runtime = prepare_gaia2_runtime(staged_gaia2, create=not args.skip_runtime)
    dataset = _prepare_dataset(reuse_source=args.reuse_source, gaia2_revision=commit)
    partition_views = None
    if args.partition != "full" or args.purpose == "trace-generation":
        partition_views = materialize_partition_views()
    filesystem = _prepare_filesystem(reuse_source=args.reuse_source)

    project_tools = {
        "python": PROJECT_VENV / "bin" / "python",
        "vllm": PROJECT_VENV / "bin" / "vllm",
        "langgraph": PROJECT_VENV / "bin" / "langgraph",
    }
    required_tools = {"python"}
    if any(
        isinstance(item, DecomposerExperiment)
        or (isinstance(item, SimpleExperiment) and not item.requires_openrouter)
        for item in experiments
    ):
        required_tools.add("vllm")
    if any(isinstance(item, DecomposerExperiment) for item in experiments):
        required_tools.add("langgraph")
    for name in required_tools:
        if not project_tools[name].is_file():
            raise FileNotFoundError(
                f"Project {name} executable is missing: {project_tools[name]}"
            )

    summaries: list[dict[str, Any]] = []
    for experiment in experiments:
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "experiment": {"name": experiment.name, "kind": experiment.kind},
            "split": SPLIT,
            "domain": DOMAIN,
            "purpose": args.purpose,
            "partition": args.partition,
            "dataset": {
                "manifest": str(dataset_manifest()),
                "dataset_revision": DATASET_REVISION,
                "rows": dataset["rows"],
                "aggregate_sha256": dataset["aggregate_sha256"],
            },
            "filesystem": {
                "manifest": str(filesystem_manifest()),
                "dataset": FILESYSTEM_DATASET_ID,
                "dataset_revision": FILESYSTEM_DATASET_REVISION,
                "file_count": filesystem["file_count"],
                "total_bytes": filesystem["total_bytes"],
                "aggregate_sha256": filesystem["aggregate_sha256"],
            },
            "partition_split": {
                "name": SPLIT_MANIFEST_NAME,
                "path": SPLIT_MANIFEST_RELPATH,
                "sha256": SPLIT_MANIFEST_SHA256,
                "views": partition_views,
            },
            "gaia2": {
                "source_repo": str(gaia2_source),
                "commit": commit,
                "staged_repo": str(staged_gaia2),
                "runtime": runtime,
            },
            "project_venv": str(PROJECT_VENV),
            "project_tools": {name: str(path) for name, path in project_tools.items()},
            "models": experiment_models(
                experiment, full_hashes=args.full_checkpoint_hashes
            ),
        }
        path = preparation_manifest(experiment)
        atomic_json(path, manifest)
        summaries.append(
            {
                "experiment": experiment.name,
                "kind": experiment.kind,
                "manifest": str(path),
            }
        )
    print(
        json.dumps(
            {
                "split": SPLIT,
                "domain": DOMAIN,
                "purpose": args.purpose,
                "partition": args.partition,
                "dataset_revision": DATASET_REVISION,
                "filesystem_dataset_revision": FILESYSTEM_DATASET_REVISION,
                "gaia2_revision": commit,
                "prepared": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    eval_parser = subparsers.add_parser("eval", help="Prepare evaluation inputs")
    eval_parser.add_argument("--split", choices=(SPLIT,), default=SPLIT)
    eval_parser.add_argument("--domain", choices=(DOMAIN,), default=DOMAIN)
    eval_parser.add_argument(
        "--purpose",
        choices=("evaluation", "trace-generation"),
        default="evaluation",
    )
    eval_parser.add_argument("--partition", choices=PARTITIONS, default="full")
    eval_parser.add_argument("--experiment", action="append")
    eval_parser.add_argument("--filter", action="append")
    eval_parser.add_argument("--gaia-repo", type=Path, default=DEFAULT_GAIA2_REPO)
    eval_parser.add_argument("--reuse-source", action="store_true")
    eval_parser.add_argument("--skip-runtime", action="store_true")
    eval_parser.add_argument("--full-checkpoint-hashes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "eval":
        return prepare_eval(args)
    raise AssertionError(f"Unhandled preparation command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
