"""Build the named Workplace Assistant SFT releases from their gym-owned specs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gyms.workplace_assistant.experiments import ARTIFACTS_ROOT  # noqa: E402

SPECS_DIR = Path(__file__).with_name("specs")
SFT_OUTPUT_ROOT = ARTIFACTS_ROOT / "datasets" / "sft"
SFT_SPECS = {
    "workplace-all-v3": "workplace_all_v3.yaml",
    "workplace-26b-nonthinking-v3": "workplace_26b_nonthinking_v3.yaml",
    "workplace-deepseek-e4b-thinking-v1": "workplace_deepseek_e4b_thinking_v1.yaml",
    "workplace-deepseek-e4b-thinking-v2-8k": (
        "workplace_deepseek_e4b_thinking_v2_8k.yaml"
    ),
    "workplace-deepseek-e4b-thinking-v2-32k": (
        "workplace_deepseek_e4b_thinking_v2_32k.yaml"
    ),
}


def prepare_sft(args: argparse.Namespace) -> int:
    from sft.builder import prepare_dataset

    spec = SPECS_DIR / SFT_SPECS[args.dataset]
    prepared = prepare_dataset(spec, args.output_root)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SFT_SPECS), required=True)
    parser.add_argument("--output-root", type=Path, default=SFT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return prepare_sft(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
