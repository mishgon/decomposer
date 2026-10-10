from __future__ import annotations

import subprocess
from pathlib import Path

from gyms.gaia2.staging import git, stage_revision


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_stage_revision_supports_submodule_worktree(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    subprocess.run(["git", "init", "--quiet", str(upstream)], check=True)
    (upstream / "payload.txt").write_text("pinned\n", encoding="utf-8")
    _git(upstream, "add", "payload.txt")
    _git(
        upstream,
        "-c",
        "user.name=Gaia2 test",
        "-c",
        "user.email=gaia2-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    commit = _git(upstream, "rev-parse", "HEAD")

    superproject = tmp_path / "superproject"
    subprocess.run(["git", "init", "--quiet", str(superproject)], check=True)
    _git(
        superproject,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        str(upstream),
        "external/gaia2",
    )
    source = superproject / "external" / "gaia2"
    assert (source / ".git").is_file()

    target = tmp_path / "staged"
    assert stage_revision(source, commit, target) == target

    assert (target / ".git").is_dir()
    assert git(target, "rev-parse", "HEAD") == commit
    assert (target / "payload.txt").read_text(encoding="utf-8") == "pinned\n"
    assert (target / ".stage_complete").read_text(encoding="utf-8") == commit + "\n"
