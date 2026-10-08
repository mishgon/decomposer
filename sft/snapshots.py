"""Immutable, digest-named snapshots of the native files an SFT adapter reads.

A build spec (spec_version 4) names each source by its snapshot digest, so a
release is pinned to exact source contents: renaming, moving, or extending the
original directory changes nothing, and any changed file fails the build. A
snapshot holds only the files its adapter reads, at the same relative paths, so
the adapter reads a snapshot exactly like the original source directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .schema import JsonObject, canonical_json, sha256_text

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_MANIFEST_NAME = "snapshot.json"
DEFAULT_SNAPSHOT_ROOT = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/datasets/sft/snapshots"
)
SNAPSHOT_REFERENCE = re.compile(r"^sha256:([0-9a-f]{64})$")
# JSON keys that hold internal model endpoints; redacted files drop them.
ENDPOINT_KEYS = frozenset(
    {"base_url", "agent_base_url", "openai_api_base", "model_proxy_unix_socket"}
)
# Environment variables whose values must never reach a snapshot: for a URL, its
# host, or its host and port when the host is a loopback address.
FORBIDDEN_ENVIRONMENT = (
    "LLM_PROXY_URL",
    "LLM_PROXY_MASTER_KEY",
    "OPENROUTER_API_KEY_DECOMPOSER",
)
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class SnapshotFile:
    """A native file an adapter reads: copied as is, or with endpoint keys dropped."""

    path: str
    redact_endpoints: bool = False


def snapshot_digest(reference: str) -> str:
    """Return the hex digest of a ``sha256:<hex>`` snapshot reference."""
    match = SNAPSHOT_REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError(f"Snapshot references must look like sha256:<64 hex>: {reference!r}")
    return match.group(1)


def snapshot_directory(root: Path, adapter: str, digest: str) -> Path:
    return root / adapter / digest[:16]


def _content_digest(adapter: str, files: dict[str, JsonObject]) -> str:
    return sha256_text(
        canonical_json(
            {
                "adapter": adapter,
                "files": {
                    path: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
                    for path, entry in files.items()
                },
            }
        )
    )


def _forbidden_values() -> dict[str, str]:
    values = {}
    for name in FORBIDDEN_ENVIRONMENT:
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        parsed = urlparse(value) if "://" in value else None
        host = parsed.hostname if parsed is not None else None
        # Traces mention loopback hosts for their own services; a loopback
        # endpoint is identified by its port.
        if parsed is not None and host in LOOPBACK_HOSTS:
            values[name] = parsed.netloc
        else:
            values[name] = host or value
    return values


def _without_endpoints(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_endpoints(item)
            for key, item in value.items()
            if key not in ENDPOINT_KEYS
        }
    if isinstance(value, list):
        return [_without_endpoints(item) for item in value]
    return value


def redact_endpoints(content: bytes) -> bytes:
    """Drop endpoint keys at any depth of a JSON document."""
    redacted = _without_endpoints(json.loads(content))
    return (json.dumps(redacted, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _forbidden_variable_in(path: Path, forbidden: dict[str, str]) -> str | None:
    """Name the environment variable whose value occurs in a file, without its value."""
    if not forbidden:
        return None
    needles = {name: value.encode("utf-8") for name, value in forbidden.items()}
    overlap = max((len(needle) for needle in needles.values()), default=1) - 1
    tail = b""
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK_BYTES):
            window = tail + chunk
            for name, needle in needles.items():
                if needle in window:
                    return name
            tail = window[-overlap:] if overlap else b""
    return None


def _relative_path(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or path != str(pure):
        raise ValueError(f"Snapshot file paths must be normalized relative paths: {path!r}")
    if path == SNAPSHOT_MANIFEST_NAME:
        raise ValueError(f"{SNAPSHOT_MANIFEST_NAME} is reserved for the snapshot manifest")
    return path


def create_snapshot(
    adapter: str,
    source_dir: str | Path,
    output_root: str | Path,
    files: Sequence[SnapshotFile],
) -> tuple[Path, JsonObject]:
    """Copy an adapter's files into ``<root>/<adapter>/<digest[:16]>``.

    Returns the snapshot directory and its manifest. An existing snapshot with the
    same contents is verified and reused, never rewritten.
    """
    source_dir = Path(source_dir).resolve()
    output_root = Path(output_root).resolve()
    paths = [_relative_path(file.path) for file in files]
    if not paths:
        raise ValueError("A snapshot needs at least one file.")
    if len(paths) != len(set(paths)):
        raise ValueError("Snapshot file paths must be unique.")
    forbidden = _forbidden_values()
    adapter_root = output_root / adapter
    adapter_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=adapter_root))
    try:
        entries: dict[str, JsonObject] = {}
        for file in sorted(files, key=lambda item: item.path):
            origin = source_dir / file.path
            if not origin.is_file():
                raise FileNotFoundError(f"Snapshot source file does not exist: {origin}")
            target = temporary_dir / file.path
            target.parent.mkdir(parents=True, exist_ok=True)
            entry: JsonObject
            if file.redact_endpoints:
                content = origin.read_bytes()
                target.write_bytes(redact_endpoints(content))
                entry = {
                    "transform": "redact_endpoints",
                    "origin_sha256": hashlib.sha256(content).hexdigest(),
                }
            else:
                shutil.copyfile(origin, target)
                entry = {"transform": "copy"}
            leaked = _forbidden_variable_in(target, forbidden)
            if leaked is not None:
                raise ValueError(
                    f"Snapshot file {file.path} contains the value of {leaked}; "
                    "redact it before snapshotting."
                )
            entry["bytes"], entry["sha256"] = _hash_file(target)
            entries[file.path] = entry
        digest = _content_digest(adapter, entries)
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "adapter": adapter,
            "digest": f"sha256:{digest}",
            "files": entries,
            "origin": {
                "path": str(source_dir),
                "host": socket.gethostname(),
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            "totals": {
                "files": len(entries),
                "bytes": sum(int(entry["bytes"]) for entry in entries.values()),
            },
        }
        (temporary_dir / SNAPSHOT_MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        snapshot_dir = snapshot_directory(output_root, adapter, digest)
        if snapshot_dir.exists():
            shutil.rmtree(temporary_dir)
            return snapshot_dir, load_snapshot(snapshot_dir, f"sha256:{digest}", adapter=adapter)
        os.rename(temporary_dir, snapshot_dir)
        return snapshot_dir, manifest
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def load_snapshot(directory: str | Path, reference: str, *, adapter: str) -> JsonObject:
    """Verify a snapshot's digest, exact file set, and every file's size and hash."""
    directory = Path(directory)
    expected = snapshot_digest(reference)
    manifest_path = directory / SNAPSHOT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snapshot manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported snapshot schema version in {manifest_path}.")
    if manifest.get("adapter") != adapter:
        raise ValueError(
            f"Snapshot {directory} belongs to adapter {manifest.get('adapter')!r}, "
            f"not {adapter!r}."
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Snapshot manifest {manifest_path} lists no files.")
    digest = _content_digest(adapter, files)
    if manifest.get("digest") != f"sha256:{digest}" or digest != expected:
        raise ValueError(
            f"Snapshot {directory} resolves to sha256:{digest}, not {reference}."
        )
    present = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if present != set(files):
        unexpected = sorted(present - set(files))
        missing = sorted(set(files) - present)
        raise ValueError(
            f"Snapshot {directory} file set differs from its manifest "
            f"(unexpected: {unexpected[:5]}, missing: {missing[:5]})."
        )
    for path, entry in sorted(files.items()):
        size, sha256 = _hash_file(directory / path)
        if size != entry.get("bytes") or sha256 != entry.get("sha256"):
            raise ValueError(f"Snapshot file {directory / path} was modified.")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    from .adapters.registry import SNAPSHOT_FILES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", required=True, choices=sorted(SNAPSHOT_FILES))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    args = parser.parse_args(argv)
    if not _forbidden_values():
        print(
            "warning: none of "
            + ", ".join(FORBIDDEN_ENVIRONMENT)
            + " is set, so the endpoint guard checks nothing; load the secrets "
            "environment first.",
            file=sys.stderr,
        )
    files = SNAPSHOT_FILES[args.adapter](args.source.resolve())
    directory, manifest = create_snapshot(
        args.adapter, args.source, args.output_root, files
    )
    print(
        json.dumps(
            {
                "adapter": args.adapter,
                "snapshot": manifest["digest"],
                "path": str(directory),
                "totals": manifest["totals"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
