"""Download one pinned WideSeek training dataset and prepare auditable task JSONL."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

DATASET = "RLinf/WideSeek-R1-train-data"
REVISION = "bd03eb2ccc171a2e1dade48c499b82fe040757f5"
SOURCES = ("width", "depth", "hybrid")


def normalize(row, source, index):
    question, answer = row.get("question"), row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{source}:{index}: missing question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{source}:{index}: missing reference answer")
    columns = row.get("unique_columns") or []
    if not isinstance(columns, list) or any(not isinstance(c, str) for c in columns):
        raise ValueError(f"{source}:{index}: invalid unique_columns")
    return {"task_id": f"{source}-{index:05d}", "source": source,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "question": question, "answer": answer, "unique_columns": columns,
            "metadata": {k: v for k, v in row.items()
                         if k not in {"question", "answer", "unique_columns"}}}


def agent_input(task):
    """References and grading metadata must never enter an agent's input."""
    return {"messages": [{"role": "user", "content": task["question"]}]}


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{source}_20k.jsonl"
    raw = output / "source.jsonl"
    with urlopen(url, timeout=120) as response, raw.open("xb") as stream:
        shutil.copyfileobj(response, stream)
    count, prompts = 0, set()
    with raw.open() as stream, (output / "tasks.jsonl").open("x") as dest:
        for index, line in enumerate(stream):
            task = normalize(json.loads(line), source, index)
            dest.write(json.dumps(task, ensure_ascii=False) + "\n")
            prompts.add(task["question_sha256"])
            count += 1
    if count != 20_000:
        raise ValueError(f"Expected 20000 examples, got {count}; dataset not marked ready")
    manifest = {"dataset": DATASET, "revision": REVISION, "source": source,
                "url": url, "examples": count, "unique_questions": len(prompts),
                "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "tasks_sha256": hashlib.sha256((output / "tasks.jsonl").read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, default="width")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("artifacts/gyms/wideseek/data") / args.source
    print(json.dumps(prepare(args.source, output), indent=2))
