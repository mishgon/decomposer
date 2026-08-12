import json
import subprocess
import sys
from pathlib import Path

import pytest

from gyms.toolathlon_gym import run


def test_configured_subagents_are_registered() -> None:
    registered = json.loads(
        (Path(run.__file__).parent / "subagents" / "langgraph.json").read_text()
    )["graphs"]

    assert {
        assistant_id for _, assistant_id, _ in run.SUBAGENT_TYPES
    } <= registered.keys()


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
