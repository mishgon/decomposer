"""Download the pinned MASBench test split, retaining original rows for audit."""
import argparse
from collections import Counter
import json
from pathlib import Path

from datasets import load_dataset
from gyms.masbench import AXES, DATASET, REVISION


def convert(row, axis, index):
    messages = json.loads(row["prompt_json"])
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError(f"Unexpected prompt format: {axis}/{index}")
    return {
        "task_id": f"{axis}-{index:04d}", "axis": axis, "value": row["value"],
        "question": messages[0]["content"],
        "ground_truth": json.loads(row["reward_model_json"])["ground_truth"],
    }


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (output / "tasks.jsonl").open("w") as tasks, (output / "source.jsonl").open("w") as source:
        for axis in AXES:
            dataset = load_dataset(DATASET, axis, split="test", revision=REVISION)
            for index, row in enumerate(dataset):
                task = convert(row, axis, index)
                tasks.write(json.dumps(task) + "\n")
                source.write(json.dumps({"task_id": task["task_id"], "row": row}) + "\n")
                counts[axis] += 1
    (output / "dataset.json").write_text(json.dumps({
        "dataset": DATASET, "revision": REVISION, "split": "test", "counts": counts,
    }, indent=2))
    print(dict(counts))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, required=True)
    prepare(cli.parse_args().output_dir)
