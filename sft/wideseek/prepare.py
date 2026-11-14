"""Combine prepared WideSeek splits and shuffle the entire collection pool."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from random import Random

from gyms.wideseek import REPO_ROOT

DEFAULT_OUTPUT = REPO_ROOT / "artifacts/sft/wideseek/data/width-depth"


def prepare(inputs, output):
    tasks, sources = [], []
    for path in inputs:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        tasks.extend(rows)
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                        "examples": len(rows)})
    if not tasks or len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Collection pool must be nonempty with unique task IDs")
    Random(42).shuffle(tasks)
    data = "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks).encode()
    manifest = {"inputs": sources, "shuffle_seed": 42, "examples": len(tasks),
                "sources": dict(Counter(task["source"] for task in tasks)),
                "tasks_sha256": hashlib.sha256(data).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    (output / "tasks.jsonl").write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="Prepared task files, e.g. width/tasks.jsonl depth/tasks.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.output), indent=2))
