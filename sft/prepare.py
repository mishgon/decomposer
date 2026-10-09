"""Build a canonical SFT release from a spec whose sources are snapshots."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .builder import prepare_dataset
from .snapshots import DEFAULT_SNAPSHOT_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help="where snapshots are looked up by digest",
    )
    args = parser.parse_args(argv)
    prepared = prepare_dataset(
        args.spec, args.output_root, snapshot_root=args.snapshot_root
    )
    print(
        json.dumps(
            {
                "dataset": prepared.manifest["dataset"],
                "release_dir": str(prepared.release_dir),
                "manifest_path": str(prepared.manifest_path),
                "filtering": prepared.manifest["filtering"],
                "records": prepared.manifest["records"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
