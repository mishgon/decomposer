"""Combine both DeepDive QA splits and shuffle across them with seed 42."""
import argparse
import json
from pathlib import Path

from gyms.deepdive import REPO_ROOT
from gyms.deepdive.prepare import SPLITS
from sft.wideseek.prepare import prepare

DEFAULT_OUTPUT = REPO_ROOT / "artifacts/sft/deepdive/data/all-qa"
DEFAULT_INPUTS = [REPO_ROOT / "artifacts/gyms/deepdive/data" / split / "tasks.jsonl"
                  for split in SPLITS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.output), indent=2))
