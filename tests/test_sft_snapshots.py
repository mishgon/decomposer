from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from sft.builder import _git_revision
from sft.snapshots import (
    SNAPSHOT_MANIFEST_NAME,
    SnapshotFile,
    create_snapshot,
    load_snapshot,
    snapshot_directory,
)


def _source(root: Path) -> Path:
    source = root / "run"
    (source / "traces").mkdir(parents=True)
    (source / "rollouts.jsonl").write_text('{"reward": 1.0}\n')
    (source / "traces" / "trace.json").write_text(
        json.dumps({"messages": [], "agent_base_url": "http://agents.internal:8000"})
    )
    (source / "unread.log").write_text("not part of the snapshot\n")
    return source


FILES = [SnapshotFile("rollouts.jsonl"), SnapshotFile("traces/trace.json")]


def test_snapshot_round_trip_copies_only_listed_files(tmp_path: Path) -> None:
    source = _source(tmp_path)
    directory, manifest = create_snapshot("nemo_gym", source, tmp_path / "snapshots", FILES)

    digest = manifest["digest"]
    assert directory == snapshot_directory(tmp_path / "snapshots", "nemo_gym", digest[7:])
    assert sorted(manifest["files"]) == ["rollouts.jsonl", "traces/trace.json"]
    assert not (directory / "unread.log").exists()
    for name in manifest["files"]:
        assert (directory / name).read_bytes() == (source / name).read_bytes()
    assert load_snapshot(directory, digest, adapter="nemo_gym")["digest"] == digest
    with pytest.raises(ValueError, match="belongs to adapter"):
        load_snapshot(directory, digest, adapter="toolathlon_gym")


def test_renamed_source_gives_the_same_snapshot(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first, first_manifest = create_snapshot("nemo_gym", source, tmp_path / "snapshots", FILES)
    renamed = source.rename(tmp_path / "renamed-run")
    second, second_manifest = create_snapshot("nemo_gym", renamed, tmp_path / "snapshots", FILES)

    assert second == first
    assert second_manifest["digest"] == first_manifest["digest"]
    # The existing snapshot is reused, so it keeps the origin it was made from.
    assert second_manifest["origin"]["path"] == str(source)


def test_modified_extra_or_missing_files_are_rejected(tmp_path: Path) -> None:
    directory, manifest = create_snapshot(
        "nemo_gym", _source(tmp_path), tmp_path / "snapshots", FILES
    )
    digest = manifest["digest"]
    with pytest.raises(ValueError, match="resolves to"):
        load_snapshot(directory, "sha256:" + "0" * 64, adapter="nemo_gym")

    (directory / "extra.txt").write_text("x")
    with pytest.raises(ValueError, match="file set differs"):
        load_snapshot(directory, digest, adapter="nemo_gym")
    (directory / "extra.txt").unlink()

    (directory / "rollouts.jsonl").write_text('{"reward": 0.0}\n')
    with pytest.raises(ValueError, match="was modified"):
        load_snapshot(directory, digest, adapter="nemo_gym")

    (directory / "rollouts.jsonl").unlink()
    with pytest.raises(ValueError, match="file set differs"):
        load_snapshot(directory, digest, adapter="nemo_gym")


def test_snapshot_paths_must_be_relative_and_unique(tmp_path: Path) -> None:
    source = _source(tmp_path)
    for files in (
        [SnapshotFile("../rollouts.jsonl")],
        [SnapshotFile("/etc/passwd")],
        [SnapshotFile(SNAPSHOT_MANIFEST_NAME)],
        [SnapshotFile("rollouts.jsonl"), SnapshotFile("rollouts.jsonl")],
    ):
        with pytest.raises(ValueError):
            create_snapshot("nemo_gym", source, tmp_path / "snapshots", files)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True)


def test_release_builds_require_committed_spec_and_no_untracked_code(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "sft" / "specs").mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.test")
    _git(repository, "config", "user.name", "Test")
    spec = repository / "sft" / "specs" / "release.yaml"
    spec.write_text("spec_version: 4\n")
    (repository / "sft" / "builder.py").write_text("")
    _git(repository, "add", "sft/builder.py")
    _git(repository, "commit", "-q", "-m", "builder")

    # The spec exists but was never committed, so it is an untracked file under sft/.
    with pytest.raises(RuntimeError, match="no untracked files"):
        _git_revision(require_clean=True, spec_path=spec, repository=repository)

    _git(repository, "add", "sft/specs/release.yaml")
    _git(repository, "commit", "-q", "-m", "spec")
    assert _git_revision(require_clean=True, spec_path=spec, repository=repository)

    (repository / "sft" / "new_adapter.py").write_text("")
    with pytest.raises(RuntimeError, match="no untracked files"):
        _git_revision(require_clean=True, spec_path=spec, repository=repository)
    (repository / "sft" / "new_adapter.py").unlink()

    outside = tmp_path / "outside.yaml"
    outside.write_text("spec_version: 4\n")
    with pytest.raises(RuntimeError, match="must be committed"):
        _git_revision(require_clean=True, spec_path=outside, repository=repository)
