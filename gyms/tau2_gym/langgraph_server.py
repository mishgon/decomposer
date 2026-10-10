"""Launch the tau2 subagent LangGraph server without disk persistence.

``langgraph dev`` keeps its in-memory store in ``.langgraph_api/`` under its working
directory: it reloads that store at startup and rewrites all of it every 10 s. Left
in ``gyms/tau2_gym/subagents`` across runs, the store grew to gigabytes and made
``threads.get_history`` (which the Decomposer's ``wait`` calls for every finished run)
take tens of seconds. ``langgraph dev`` cannot switch this off: the CLI drops the
``disable_persistence`` key from ``langgraph.json``, and ``run_server`` then sets
``LANGGRAPH_DISABLE_FILE_PERSISTENCE=false`` itself. So this calls ``run_server``
directly, as ``gyms/gaia2/langgraph_server.py`` does, from a per-run directory.

The graphs come from the committed ``subagents/langgraph.json``. Their file paths
(``./graph.py:fn``) become module paths (``gyms.tau2_gym.subagents.graph:fn``), which import through
``PYTHONPATH`` from any working directory.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SUBAGENTS_PACKAGE = "gyms.tau2_gym.subagents"
SOURCE_CONFIG = Path(__file__).resolve().parent / "subagents" / "langgraph.json"
RUNTIME_CONFIG_NAME = "langgraph.json"


def runtime_config(source: Path = SOURCE_CONFIG) -> dict[str, Any]:
    """The server config for one run: the committed graphs, persistence disabled."""
    committed = json.loads(source.read_text(encoding="utf-8"))
    graphs: dict[str, str] = {}
    for name, spec in committed["graphs"].items():
        path, _, function = spec.rpartition(":")
        if not path.endswith(".py") or not function:
            raise ValueError(f"{source}: graph {name!r} is not a ./<file>.py:<function> spec: {spec!r}")
        module = path.removeprefix("./").removesuffix(".py").replace("/", ".")
        graphs[name] = f"{SUBAGENTS_PACKAGE}.{module}:{function}"
    return {
        "graphs": graphs,
        "python_version": committed.get("python_version", "3.12"),
        "disable_persistence": True,
    }


def write_runtime_config(directory: Path, source: Path = SOURCE_CONFIG) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUNTIME_CONFIG_NAME
    path.write_text(json.dumps(runtime_config(source), indent=2) + "\n", encoding="utf-8")
    return path


def load_runtime_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("disable_persistence") is not True:
        raise ValueError("tau2 LangGraph file persistence must be disabled")
    graphs = config.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise ValueError("tau2 LangGraph runtime requires at least one graph")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--n-jobs-per-worker", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_runtime_config(args.config)

    from langgraph_api.cli import run_server

    run_server(
        host=args.host,
        port=args.port,
        reload=False,
        graphs=config["graphs"],
        n_jobs_per_worker=args.n_jobs_per_worker,
        open_browser=False,
        disable_persistence=True,
        # As `langgraph dev` ran these graphs before.
        allow_blocking=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
