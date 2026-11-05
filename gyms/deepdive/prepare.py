"""Pin DeepDive QA splits and preserve every task for later solvability filtering."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

from gyms.deepdive import REPO_ROOT
from gyms.wideseek.prepare import normalize as normalize_qa

DATASET = "zai-org/DeepDive"
REVISION = "36b28945eaf271e64003e7fcd4ddb81af0828eb5"
SPLITS = {"qa_rl": 2234, "qa_sft": 1016}


def normalize(row, split):
    if type(row.get("id")) is not int or row["id"] < 0:
        raise ValueError("DeepDive task requires a nonnegative integer id")
    task = normalize_qa({"question": row["question"], "answer": row["answer"]},
                        f"deepdive-{split}", row["id"])
    task["metadata"] = {"dataset": DATASET, "revision": REVISION,
                        "split": split, "original_id": row["id"]}
    return task


def prepare(split, output):
    import pyarrow.parquet as pq

    expected = SPLITS[split]
    output.mkdir(parents=True, exist_ok=False)
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{split}-00000-of-00001.parquet"
    raw = output / "source.parquet"
    with urlopen(url, timeout=120) as response, raw.open("xb") as stream:
        shutil.copyfileobj(response, stream)
    tasks = [normalize(row, split) for row in pq.read_table(raw).to_pylist()]
    if len(tasks) != expected or len({t["task_id"] for t in tasks}) != expected:
        raise ValueError(f"Expected {expected} distinct {split} tasks")
    data = output / "tasks.jsonl"
    data.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))
    manifest = {"dataset": DATASET, "revision": REVISION, "split": split,
                "url": url, "examples": len(tasks), "filtered": False,
                "unique_questions": len({t["question_sha256"] for t in tasks}),
                "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "tasks_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="qa_rl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or REPO_ROOT / "artifacts/gyms/deepdive/data" / args.split
    print(json.dumps(prepare(args.split, output), indent=2))
