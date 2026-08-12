"""CLI for building immutable, versioned Decomposer SFT datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .builder import prepare_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an immutable canonical Decomposer SFT dataset release."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    prepared = prepare_dataset(args.spec, args.output_root)
    print(
        json.dumps(
            {
                "dataset": prepared.manifest["dataset"],
                "release_dir": str(prepared.release_dir),
                "manifest_path": str(prepared.manifest_path),
                "filtering": prepared.manifest["filtering"],
                "records": prepared.manifest["records"],
                "split": {
                    key: value
                    for key, value in prepared.manifest["split"].items()
                    if key != "validation_group_ids"
                },
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
