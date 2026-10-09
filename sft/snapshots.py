"""Immutable, digest-named snapshots of the native files an SFT adapter reads.

A build spec (spec_version 4) names each source by its snapshot digest, so a
release is pinned to exact source contents: renaming, moving, or extending the
original directory changes nothing, and any changed file fails the build. A
snapshot holds only the files its adapter reads, at the same relative paths, so
the adapter reads a snapshot exactly like the original source directory. Files
are copied byte for byte: a snapshot never changes a trace.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from .schema import JsonObject, canonical_json, sha256_file, sha256_text

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_MANIFEST_NAME = "snapshot.json"
DEFAULT_SNAPSHOT_ROOT = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/datasets/sft/snapshots"
)


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


def create_snapshot(
    adapter: str,
    source_dir: str | Path,
    output_root: str | Path,
    files: Sequence[str],
) -> tuple[Path, JsonObject]:
    """Copy an adapter's files into ``<root>/<adapter>/<digest[:16]>``.

    Returns the snapshot directory and its manifest. An existing snapshot with the
    same contents is verified and reused, never rewritten.
    """
    source_dir = Path(source_dir).resolve()
    output_root = Path(output_root).resolve()
    if not files:
        raise ValueError("A snapshot needs at least one file.")
    adapter_root = output_root / adapter
    adapter_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=adapter_root))
    try:
        entries: dict[str, JsonObject] = {}
        for path in sorted(files):
            origin = source_dir / path
            if not origin.is_file():
                raise FileNotFoundError(f"Snapshot source file does not exist: {origin}")
            target = temporary_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
            entries[path] = {
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
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
    digest = f"sha256:{_content_digest(adapter, files)}"
    if manifest.get("digest") != digest or digest != reference:
        raise ValueError(f"Snapshot {directory} resolves to {digest}, not {reference}.")
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
        file_path = directory / path
        if (
            file_path.stat().st_size != entry.get("bytes")
            or sha256_file(file_path) != entry.get("sha256")
        ):
            raise ValueError(f"Snapshot file {file_path} was modified.")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    from .adapters.registry import SNAPSHOT_FILES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", required=True, choices=sorted(SNAPSHOT_FILES))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    args = parser.parse_args(argv)
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
