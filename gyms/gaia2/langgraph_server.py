"""Launch the GAIA2 LangGraph runtime without disk persistence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def load_runtime_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("disable_persistence") is not True:
        raise ValueError("GAIA2 LangGraph file persistence must be disabled")
    graphs = config.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise ValueError("GAIA2 LangGraph runtime requires at least one graph")
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
        allow_blocking=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
