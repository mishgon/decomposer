from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

ARTIFACTS_ROOT = Path(__file__).resolve().parents[2] / "artifacts"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAIA2_REPO = PROJECT_ROOT / "external" / "gaia2"

DATASET_ID = "meta-agents-research-environments/gaia2"
DATASET_REVISION = "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99"
FILESYSTEM_DATASET_ID = "meta-agents-research-environments/gaia2_filesystem"
FILESYSTEM_DATASET_REVISION = "132e26376f5e963bb59f64bcccdd02188cb08dee"
GAIA2_REVISION = "221a9de19275c8545bc70236874d40f3ce4d97a1"
SPLIT = "validation"

Gaia2Domain = Literal["execution", "search"]


@dataclass(frozen=True)
class Gaia2DomainSpec:
    name: Gaia2Domain
    scenario_count: int
    dataset_aggregate_sha256: str


EXECUTION_DOMAIN = Gaia2DomainSpec(
    name="execution",
    scenario_count=160,
    dataset_aggregate_sha256=(
        "600818deac34268a261cd846ef3019c6a4c87b068dccfea5e006cd72fe112518"
    ),
)
SEARCH_DOMAIN = Gaia2DomainSpec(
    name="search",
    scenario_count=160,
    dataset_aggregate_sha256=(
        "8cc67f8334d72e1dd760685cf4c5e274cd225b5cc63369fdabf1d5e3c84ee6df"
    ),
)
DOMAIN_SPECS: dict[Gaia2Domain, Gaia2DomainSpec] = {
    spec.name: spec for spec in (EXECUTION_DOMAIN, SEARCH_DOMAIN)
}
DOMAINS = tuple(DOMAIN_SPECS)
DOMAIN: Gaia2Domain = "execution"

SCENARIO_COUNT = EXECUTION_DOMAIN.scenario_count


def get_domain_spec(domain: str) -> Gaia2DomainSpec:
    try:
        return DOMAIN_SPECS[domain]  # type: ignore[index]
    except KeyError as error:
        expected = ", ".join(DOMAINS)
        raise ValueError(
            f"Unknown Gaia2 domain {domain!r}; expected one of: {expected}"
        ) from error


DATA_ROOT = ARTIFACTS_ROOT / "data" / "gaia2"
RESULTS_ROOT = ARTIFACTS_ROOT / "evals" / "gaia2"
GAIA2_VENV_ROOT = DATA_ROOT / "venvs"
UV_CACHE = Path(os.environ.get("UV_CACHE_DIR", Path.home() / ".cache" / "uv"))
UV_BIN = Path.home() / ".local" / "bin" / "uv"
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
JUDGE_MODEL = "openai/nvidia/Llama-3.3-70B-Instruct-FP8"
SCENARIO_TIMEOUT_SECONDS = 3600


def gaia2_lock_hash(gaia2_root: Path) -> str:
    return hashlib.sha256((gaia2_root / "uv.lock").read_bytes()).hexdigest()[:16]


def gaia2_venv(gaia2_root: Path) -> Path:
    return GAIA2_VENV_ROOT / gaia2_lock_hash(gaia2_root)


def dataset_revision_root(domain: Gaia2Domain = DOMAIN) -> Path:
    """Return the immutable source root for one domain.

    Execution keeps its historical layout. Additional domains use an isolated
    subtree so preparing them cannot rewrite the completed execution source.
    """

    spec = get_domain_spec(domain)
    root = DATA_ROOT / DATASET_REVISION
    return root if spec.name == DOMAIN else root / "domains" / spec.name


def dataset_root(domain: Gaia2Domain = DOMAIN) -> Path:
    """Root passed to ARE with the selected ``--config``."""

    return dataset_revision_root(domain) / SPLIT


def scenario_dir(domain: Gaia2Domain = DOMAIN) -> Path:
    return dataset_root(domain) / domain


def dataset_manifest(domain: Gaia2Domain = DOMAIN) -> Path:
    return dataset_revision_root(domain) / "dataset_manifest.json"


def filesystem_revision_root() -> Path:
    return DATA_ROOT / "filesystem" / FILESYSTEM_DATASET_REVISION


def filesystem_dir() -> Path:
    return filesystem_revision_root() / "demo_filesystem"


def filesystem_manifest() -> Path:
    return filesystem_revision_root() / "filesystem_manifest.json"


def preparation_manifest(domain: Gaia2Domain = DOMAIN) -> Path:
    return DATA_ROOT / "manifests" / f"{domain}.json"


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
    domain: Gaia2Domain = DOMAIN,
) -> dict[str, Any]:
    """Write one immutable dataset revision atomically."""

    spec = get_domain_spec(domain)

    if destination_revision_root.exists():
        return validate_materialized_dataset(destination_revision_root, domain=domain)

    destination_revision_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_revision_root.with_name(
        f".{destination_revision_root.name}.tmp.{os.getpid()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    scenario_directory = temporary / SPLIT / spec.name
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
        if len(entries) != spec.scenario_count:
            raise ValueError(
                f"Expected {spec.scenario_count} Gaia2 {spec.name} scenarios, "
                f"found {len(entries)}"
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
            "domain": spec.name,
            "split": SPLIT,
            "rows": len(entries),
            "aggregate_sha256": aggregate.hexdigest(),
            "gaia2_revision": gaia2_revision,
            "scenario_directory": str(destination_revision_root / SPLIT / spec.name),
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
        return validate_materialized_dataset(destination_revision_root, domain=domain)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_materialized_dataset(
    revision_root: Path,
    *,
    domain: Gaia2Domain = DOMAIN,
) -> dict[str, Any]:
    spec = get_domain_spec(domain)
    manifest_path = revision_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared Gaia2 manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "domain": spec.name,
        "split": SPLIT,
        "rows": spec.scenario_count,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Prepared Gaia2 manifest mismatch: {mismatches}")
    scenario_directory = revision_root / SPLIT / spec.name
    if Path(manifest.get("scenario_directory", "")) != scenario_directory:
        raise ValueError("Prepared Gaia2 manifest points at an unexpected directory")
    entries = manifest.get("scenarios")
    if not isinstance(entries, list) or len(entries) != spec.scenario_count:
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


def load_huggingface_rows(
    *, cache_dir: Path, domain: Gaia2Domain = DOMAIN
) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return load_dataset(
        DATASET_ID,
        domain,
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


def git_revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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


def prepare_gaia2_runtime(gaia2_source: Path, *, create: bool) -> dict[str, Any]:
    uv = _uv_executable()
    venv = gaia2_venv(gaia2_source)
    benchmark = venv / "bin" / "are-benchmark"
    if create:
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
                str(gaia2_source),
                "--frozen",
                "--python",
                "3.12",
            ],
            check=True,
            env=env,
        )
        # FastAPI is pinned in ARE's lockfile but belongs to its optional GUI extras.
        subprocess.run(
            [
                str(uv),
                "pip",
                "install",
                "--no-deps",
                "--python",
                str(venv / "bin" / "python"),
                "fastapi==0.116.1",
            ],
            check=True,
            env=env,
        )
    if not benchmark.is_file():
        raise FileNotFoundError(
            f"Prepared Gaia2 executable is missing: {benchmark}; rerun without --skip-runtime"
        )
    lock = gaia2_source / "uv.lock"
    return {
        "path": str(venv),
        "python": str(venv / "bin" / "python"),
        "are_benchmark": str(benchmark),
        "uv_lock_sha256": sha256_file(lock),
        "uv_bin": str(uv),
        "uv_sha256": sha256_file(uv),
    }


def _prepare_dataset(
    *, reuse_source: bool, gaia2_revision: str, domain: str
) -> dict[str, Any]:
    spec = get_domain_spec(domain)
    revision_root = dataset_revision_root(spec.name)
    if revision_root.exists():
        dataset = validate_materialized_dataset(revision_root, domain=spec.name)
    else:
        if reuse_source:
            raise FileNotFoundError(
                f"Prepared Gaia2 data is missing: {dataset_manifest(spec.name)}"
            )
        rows = load_huggingface_rows(cache_dir=HF_HOME / "datasets", domain=spec.name)
        dataset = write_materialized_dataset(
            rows,
            revision_root,
            gaia2_revision=gaia2_revision,
            domain=spec.name,
        )
    if dataset.get("aggregate_sha256") != spec.dataset_aggregate_sha256:
        raise ValueError(
            f"Prepared Gaia2 {spec.name} checksum does not match the pinned "
            "dataset revision"
        )
    return dataset


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
    spec = get_domain_spec(args.domain)
    gaia2_source = DEFAULT_GAIA2_REPO
    commit = git_revision(gaia2_source)
    if commit != GAIA2_REVISION:
        raise ValueError(
            f"Gaia2 revision resolved to {commit}, expected {GAIA2_REVISION}"
        )
    runtime = prepare_gaia2_runtime(gaia2_source, create=not args.skip_runtime)
    dataset = _prepare_dataset(
        reuse_source=args.reuse_source,
        gaia2_revision=commit,
        domain=spec.name,
    )
    filesystem = _prepare_filesystem(reuse_source=args.reuse_source)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "split": SPLIT,
        "domain": spec.name,
        "dataset": {
            "manifest": str(dataset_manifest(spec.name)),
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
        "gaia2": {
            "source_repo": str(gaia2_source),
            "commit": commit,
            "runtime": runtime,
        },
    }
    path = preparation_manifest(spec.name)
    atomic_json(path, manifest)
    print(json.dumps({"manifest": str(path), **manifest}, indent=2))

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    eval_parser = subparsers.add_parser("eval", help="Prepare evaluation inputs")
    eval_parser.add_argument("--domain", choices=DOMAINS, default=DOMAIN)
    eval_parser.add_argument("--reuse-source", action="store_true")
    eval_parser.add_argument("--skip-runtime", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "eval":
        return prepare_eval(args)
    raise AssertionError(f"Unhandled preparation command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
