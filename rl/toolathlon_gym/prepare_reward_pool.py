"""Freeze a task pool from one completed five-attempt Gym evaluation."""
import argparse
import hashlib
import json
from pathlib import Path


def build_pool(manifest):
    if manifest["config"]["repetitions"] != 5:
        raise ValueError("Expected five attempts per task")
    tasks = []
    for name in sorted(manifest["tasks"]):
        rows = [manifest["episodes"].get(f"{name}/rep-{i:03d}", {}) for i in range(1, 6)]
        if any(r.get("status") not in {"completed", "infrastructure_error"} for r in rows):
            raise ValueError(f"Unfinished task: {name}")
        scores = [r.get("partial_score") if r["status"] == "completed" else None for r in rows]
        successes = sum(s is not None and s >= .9 for s in scores)
        if 0 < successes < 5:
            tasks.append({"task_id": name, "successes_ge90": successes, "partial_scores": scores,
                          "infrastructure_errors": sum(r["status"] == "infrastructure_error" for r in rows)})
    return {"threshold": .9, "attempts": 5, "selection": "1 to 4 successes; infrastructure/missing scores count as failures",
            "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    pool = build_pool(json.loads(raw))
    pool["source_run"] = args.manifest.parent.name
    pool["source_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(pool, indent=2) + "\n")
    print(f"Saved {len(pool['tasks'])} tasks to {args.output}")


if __name__ == "__main__":
    main()
