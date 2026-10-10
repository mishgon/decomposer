from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from gyms.tau2_gym import langgraph_server

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_config_serves_every_committed_graph_without_persistence(tmp_path: Path) -> None:
    committed = json.loads(langgraph_server.SOURCE_CONFIG.read_text())["graphs"]
    path = langgraph_server.write_runtime_config(tmp_path / "langgraph")
    config = langgraph_server.load_runtime_config(path)
    assert config["disable_persistence"] is True
    assert set(config["graphs"]) == set(committed)
    sys.path.insert(0, str(REPO_ROOT / "external" / "Gym"))
    for name, spec in config["graphs"].items():
        module, _, function = spec.partition(":")
        assert module == "gyms.tau2_gym.subagents.graph"
        assert committed[name] == f"./graph.py:{function}"
        assert callable(getattr(importlib.import_module(module), function))


def test_a_persistent_config_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "langgraph.json"
    path.write_text(json.dumps({"graphs": {"g": "m:f"}}))
    with pytest.raises(ValueError, match="persistence must be disabled"):
        langgraph_server.load_runtime_config(path)
    path.write_text(json.dumps({"graphs": {}, "disable_persistence": True}))
    with pytest.raises(ValueError, match="at least one graph"):
        langgraph_server.load_runtime_config(path)
