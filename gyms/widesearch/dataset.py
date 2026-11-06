"""Download a pinned WideSearch snapshot and export English questions only."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from gyms.browsecomp import REPO_ROOT

REVISION = "6531a7e5b497d44c8912407e0cb3dc95bd98cc09"
SOURCE = REPO_ROOT / "artifacts/data/widesearch/source"
TASKS = REPO_ROOT / "artifacts/data/widesearch/tasks.jsonl"


def prepare(source=SOURCE, output=TASKS):
    rows = [json.loads(line) for line in (source / "widesearch.jsonl").read_text().splitlines() if line.strip()]
    tasks = [{"task_id": row["instance_id"], "question": row["query"]}
             for row in rows if row["language"] == "en"]
    if len(tasks) != 100 or len({t["task_id"] for t in tasks}) != 100:
        raise ValueError("Expected 100 unique English WideSearch tasks")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks))
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=TASKS)
    args = parser.parse_args()
    snapshot_download("ByteDance-Seed/WideSearch", repo_type="dataset", revision=REVISION,
                      local_dir=args.source, allow_patterns=["widesearch.jsonl", "widesearch_gold/*", "README.md"])
    prepare(args.source, args.output)
