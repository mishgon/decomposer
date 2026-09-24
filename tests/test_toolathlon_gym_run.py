import json
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


def test_main_rejects_path_traversal(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", "../finalpool"])

    with pytest.raises(ValueError, match="Unknown Toolathlon task"):
        run.main()
