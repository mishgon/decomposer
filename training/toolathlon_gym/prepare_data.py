"""Freeze a task-level RL split without importing CUDA, veRL or the Gym runner."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def make_split(tasks_dir: Path, validation_tasks: int, seed: int) -> dict:
    tasks = sorted(
        path.name for path in tasks_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not tasks or not 0 <= validation_tasks < len(tasks):
        raise ValueError("Need at least one training task and a valid validation count")
    missing = [name for name in tasks if not (tasks_dir / name / "task_config.json").is_file()]
    if missing:
        raise ValueError(f"Missing task_config.json: {missing}")
    ranked = sorted(tasks, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    validation = set(ranked[:validation_tasks])
    return {
        "seed": seed,
        "train": [name for name in tasks if name not in validation],
        "validation": [name for name in tasks if name in validation],
        "task_config_sha256": {
            name: hashlib.sha256((tasks_dir / name / "task_config.json").read_bytes()).hexdigest()
            for name in tasks
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-root", type=Path, default=Path("external/toolathlon_gym"))
    parser.add_argument("--validation-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_split(args.gym_root / "tasks/finalpool", args.validation_tasks, args.seed)
    result["gym_revision"] = subprocess.check_output(
        ["git", "-C", str(args.gym_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    result["gym_dirty"] = bool(subprocess.check_output(
        ["git", "-C", str(args.gym_root), "status", "--porcelain"], text=True,
    ).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"{len(result['train'])} training / {len(result['validation'])} validation tasks: {args.output}")


if __name__ == "__main__":
    main()
