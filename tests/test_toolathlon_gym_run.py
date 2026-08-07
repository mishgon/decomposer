import subprocess
import sys

import pytest

from gyms.toolathlon_gym import run


def test_subagent_types() -> None:
    type_ids = [subagent_type_id for subagent_type_id, _, _ in run.SUBAGENT_TYPES]

    assert type_ids == [
        "tiny_thinking",
        "tiny_non_thinking",
        "small_thinking",
        "small_non_thinking",
        "medium_thinking",
        "medium_non_thinking",
        "large_thinking",
        "large_non_thinking",
    ]


def test_docker(monkeypatch) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "output", "")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    result = run._docker("ps", check=False)

    assert result.stdout == "output"
    assert calls == [
        ((["docker", "ps"],), {"check": False, "capture_output": True, "text": True})
    ]


def test_main_rejects_path_traversal(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", "../finalpool"])

    with pytest.raises(ValueError, match="Unknown Toolathlon task"):
        run.main()
