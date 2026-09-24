"""Build a canonical SFT release from a spec and explicit source locators."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .builder import prepare_dataset


def _source_override(value: str) -> tuple[str, Path]:
    source_id, separator, raw_path = value.partition("=")
    if not separator or not source_id or not raw_path:
        raise argparse.ArgumentTypeError("source overrides must use SOURCE_ID=PATH")
    return source_id, Path(raw_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        type=_source_override,
        metavar="SOURCE_ID=PATH",
    )
    args = parser.parse_args(argv)
    source_paths: dict[str, Path] = {}
    for source_id, path in args.source:
        if source_id in source_paths:
            parser.error(f"duplicate --source override for {source_id!r}")
        source_paths[source_id] = path
    prepared = prepare_dataset(
        args.spec,
        args.output_root,
        source_paths=source_paths,
    )
    print(
        json.dumps(
            {
                "dataset": prepared.manifest["dataset"],
                "release_dir": str(prepared.release_dir),
                "manifest_path": str(prepared.manifest_path),
                "filtering": prepared.manifest["filtering"],
                "records": prepared.manifest["records"],
                "tokenization": prepared.manifest.get("tokenization"),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
