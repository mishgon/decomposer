"""Git revision staging helpers shared by Gaia2 preparation and MLSpace."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def resolve_revision(repo: Path, revision: str) -> str:
    return git(repo, "rev-parse", f"{revision}^{{commit}}")


def tracked_dirty(repo: Path) -> list[str]:
    return git(repo, "status", "--porcelain", "--untracked-files=no").splitlines()


def stage_revision(repo: Path, revision: str, target: Path) -> Path:
    """Materialize a clean Git revision without reading the live worktree."""

    commit = resolve_revision(repo, revision)
    marker = target / ".stage_complete"
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != commit:
            raise RuntimeError(f"Staged revision marker does not match {target}")
        return target
    if target.exists():
        raise RuntimeError(f"Incomplete staging directory exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        git_metadata = repo / ".git"
        if git_metadata.is_file():
            # A submodule worktree stores only a relative gitdir pointer in
            # ``.git``. Copying that file would leave a broken pointer once
            # the checkout is moved under the artifact root, so materialize a
            # standalone local clone instead.
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--local",
                    "--no-hardlinks",
                    "--no-checkout",
                    str(repo),
                    str(temporary),
                ],
                check=True,
            )
        else:
            subprocess.run(
                ["cp", "-a", str(git_metadata), str(temporary)], check=True
            )
        subprocess.run(
            ["git", "-C", str(temporary), "reset", "--hard", commit], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(temporary),
                "submodule",
                "update",
                "--init",
                "--recursive",
            ],
            check=True,
        )
        (temporary / ".stage_complete").write_text(commit + "\n", encoding="utf-8")
        try:
            os.rename(temporary, target)
        except FileExistsError:
            shutil.rmtree(temporary)
            if not marker.is_file() or marker.read_text().strip() != commit:
                raise
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
