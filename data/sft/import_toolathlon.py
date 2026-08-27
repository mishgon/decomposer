"""Securely import one generated Toolathlon-Gym run for SFT preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ARTIFACTS_ROOT = Path(
    os.environ.get(
        "DECOMPOSER_ARTIFACTS_ROOT",
        "/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts",
    )
)
IMPORT_ROOT = ARTIFACTS_ROOT / "evaluation" / "data" / "toolathlon_gym" / "imports"
IMPORT_SCHEMA_VERSION = 1
DEFAULT_ARCHIVE_PREFIX = "toolathlon_gym"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError(f"Invalid Toolathlon run ID: {run_id!r}")


def _safe_relative_path(value: str, *, description: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in value
    ):
        raise ValueError(f"Unsafe {description}: {value!r}")
    return path


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in name
    ):
        raise ValueError(f"Unsafe Toolathlon archive member: {name!r}")
    return path


def _selected_relative_path(
    path: PurePosixPath,
    run_id: str,
    archive_prefix: PurePosixPath,
) -> Path | None:
    prefix_parts = archive_prefix.parts
    if path.parts[: len(prefix_parts)] != prefix_parts:
        return None
    parts = path.parts[len(prefix_parts) :]
    if parts == ("runs", run_id, "manifest.json"):
        return Path(*parts)
    if len(parts) != 4:
        return None
    kind, _task, episode_id, filename = parts
    if not episode_id.startswith(run_id + "-"):
        return None
    if kind == "traces" and filename in {"trace.json", "runtime.json"}:
        return Path(*parts)
    if kind == "evals" and filename == "result.json":
        return Path(*parts)
    return None


def _manifest_relative_path(value: str, manifest_path: Path) -> Path:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.name == "import_manifest.json"
        or ".." in relative.parts
        or "\\" in value
    ):
        raise ValueError(f"Unsafe imported-file path in {manifest_path}: {value!r}")
    return Path(*relative.parts)


def _validate_existing_import(
    destination: Path,
    archive_sha256: str,
    run_id: str,
    archive_prefix: PurePosixPath,
) -> None:
    manifest_path = destination / "import_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(
            f"Import destination exists without a manifest: {destination}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != IMPORT_SCHEMA_VERSION
        or manifest.get("run_id") != run_id
        or (manifest.get("archive") or {}).get("sha256") != archive_sha256
        or (manifest.get("archive") or {}).get("prefix")
        != archive_prefix.as_posix()
    ):
        raise FileExistsError(
            f"Import destination belongs to different source content: {destination}"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"Invalid Toolathlon import manifest: {manifest_path}")
    tracked_files: set[str] = set()
    for relative, metadata in files.items():
        if not isinstance(relative, str):
            raise ValueError(f"Invalid imported-file entry in {manifest_path}")
        safe_relative = _manifest_relative_path(relative, manifest_path)
        path = destination / safe_relative
        if (
            not isinstance(metadata, Mapping)
            or not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(destination)
            or path.stat().st_size != metadata.get("bytes")
            or sha256_file(path) != metadata.get("sha256")
        ):
            raise ValueError(f"Existing Toolathlon import changed: {path}")
        tracked_files.add(safe_relative.as_posix())
    actual_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != tracked_files:
        raise ValueError(
            "Existing Toolathlon import file set changed: "
            f"untracked={sorted(actual_files - tracked_files)}, "
            f"missing={sorted(tracked_files - actual_files)}"
        )


def import_archive(
    archive: Path,
    run_id: str,
    output_root: Path,
    *,
    expected_sha256: str | None = None,
    archive_prefix: str = DEFAULT_ARCHIVE_PREFIX,
) -> Path:
    """Securely materialize one run's SFT-relevant files from an archive."""
    archive = archive.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    _validate_run_id(run_id)
    safe_archive_prefix = _safe_relative_path(
        archive_prefix,
        description="Toolathlon archive prefix",
    )
    if not archive.is_file():
        raise FileNotFoundError(f"Toolathlon archive does not exist: {archive}")
    archive_sha256 = sha256_file(archive)
    if expected_sha256 is not None and archive_sha256 != expected_sha256.lower():
        raise ValueError(
            f"Archive SHA-256 mismatch: expected {expected_sha256}, got {archive_sha256}"
        )

    destination = output_root / run_id
    if destination.exists():
        _validate_existing_import(
            destination,
            archive_sha256,
            run_id,
            safe_archive_prefix,
        )
        return destination

    selected: dict[Path, bytes] = {}
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as file:
        for member in file:
            path = _safe_member_path(member.name)
            if member.name in seen:
                raise ValueError(f"Duplicate Toolathlon archive member: {member.name}")
            seen.add(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(
                    f"Unsupported Toolathlon archive member type: {member.name}"
                )
            relative = _selected_relative_path(path, run_id, safe_archive_prefix)
            if relative is None:
                continue
            if relative in selected:
                raise ValueError(
                    f"Duplicate normalized Toolathlon archive member: {member.name}"
                )
            stream = file.extractfile(member)
            if stream is None:
                raise ValueError(
                    f"Cannot read Toolathlon archive member: {member.name}"
                )
            selected[relative] = stream.read()

    run_manifest_relative = Path("runs") / run_id / "manifest.json"
    if run_manifest_relative not in selected:
        raise ValueError(f"Archive has no manifest for Toolathlon run {run_id}")
    try:
        run_manifest = json.loads(selected[run_manifest_relative])
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid run manifest for {run_id}: {error}") from error
    if not isinstance(run_manifest, dict) or run_manifest.get("run_id") != run_id:
        raise ValueError(f"Archive run manifest does not match {run_id}")
    if not any(path.name == "trace.json" for path in selected):
        raise ValueError(f"Archive contains no traces for Toolathlon run {run_id}")

    files = {
        relative.as_posix(): {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for relative, content in sorted(selected.items(), key=lambda item: str(item[0]))
    }
    import_manifest = {
        "schema_version": IMPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
            "prefix": safe_archive_prefix.as_posix(),
        },
        "files": files,
        "counts": {
            "traces": sum(path.name == "trace.json" for path in selected),
            "runtimes": sum(path.name == "runtime.json" for path in selected),
            "evaluations": sum(path.name == "result.json" for path in selected),
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_root))
    try:
        for relative, content in selected.items():
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (temporary / "import_manifest.json").write_text(
            json.dumps(import_manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(
                f"Toolathlon import created concurrently: {destination}"
            )
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def prepare_import(args: argparse.Namespace) -> int:
    destination = import_archive(
        args.archive,
        args.run_id,
        args.output_root,
        expected_sha256=args.expected_sha256,
        archive_prefix=args.archive_prefix,
    )
    manifest = json.loads(
        (destination / "import_manifest.json").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "source_dir": str(destination),
                "run_id": manifest["run_id"],
                "archive_sha256": manifest["archive"]["sha256"],
                "counts": manifest["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--archive-prefix",
        default=DEFAULT_ARCHIVE_PREFIX,
        help="Directory inside the archive that contains runs/, traces/, and evals/.",
    )
    parser.add_argument("--output-root", type=Path, default=IMPORT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return prepare_import(args)


if __name__ == "__main__":
    raise SystemExit(main())
