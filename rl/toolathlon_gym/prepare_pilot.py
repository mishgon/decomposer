"""Prepare a fixed RL split, hash-selected or explicitly supplied by the caller."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from rl.toolathlon_gym.prepare_data import make_split


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-tasks", type=int, default=16)
    parser.add_argument("--validation-tasks", type=int, default=8)
    parser.add_argument("--train-task", action="append")
    parser.add_argument("--validation-task", action="append")
    parser.add_argument("--evaluate-training-tasks", action="store_true",
                        help="Evaluate every training task; no held-out validation set")
    parser.add_argument("--exclude-task", action="append", default=[])
    parser.add_argument("--task-pool", type=Path, help="JSON task pool; replaces the local-server-only filter")
    parser.add_argument("--groups-per-task", type=int, default=1)
    args = parser.parse_args()
    if args.groups_per_task < 1:
        parser.error("groups-per-task must be positive")
    root = Path("external/toolathlon_gym")
    tasks = root / "tasks/finalpool"
    local_servers = {"canvas", "notion", "google_sheet", "excel", "filesystem", "word",
                     "emails", "google_calendar", "woocommerce", "google_forms", "pptx"}
    eligible = [p.parent.name for p in tasks.glob("*/task_config.json")
                if set(json.loads(p.read_text())["needed_mcp_servers"]) <= local_servers]
    if args.task_pool:
        pool = json.loads(args.task_pool.read_text())
        eligible = [row["task_id"] for row in pool["tasks"]]
        if len(eligible) != len(set(eligible)) or any(not (tasks / name / "task_config.json").exists() for name in eligible):
            parser.error("Task pool contains duplicate or unavailable tasks")
    eligible.sort(key=lambda name: hashlib.sha256(f"{args.seed}:{name}".encode()).hexdigest())
    eligible = [name for name in eligible if name not in args.exclude_task]
    if args.evaluate_training_tasks:
        if args.train_task or args.validation_task:
            parser.error("Use split sizes with --evaluate-training-tasks")
        if not 1 <= args.train_tasks <= len(eligible):
            parser.error("Invalid training task count")
        selected = {"train": sorted(eligible[:args.train_tasks])}
        selection = "Hash-selected training tasks; evaluation reuses all training tasks, no held-out validation"
    elif args.train_task is not None or args.validation_task is not None:
        if not args.train_task or not args.validation_task:
            parser.error("Explicit selection requires both training and validation tasks")
        selected = {"train": sorted(set(args.train_task)), "validation": sorted(set(args.validation_task))}
        if set(selected["train"]) & set(selected["validation"]):
            parser.error("Training and validation tasks must be disjoint")
        if not set(selected["train"] + selected["validation"]) <= set(eligible):
            parser.error("Selected tasks must be available in the local-fixture pool")
        selection = "Explicit caller selection; not a reward-blind or representative benchmark sample"
    else:
        count = args.train_tasks + args.validation_tasks
        if min(args.train_tasks, args.validation_tasks) < 1 or count > len(eligible):
            raise ValueError("Invalid pilot split sizes")
        selected = {"train": sorted(eligible[:args.train_tasks]),
                    "validation": sorted(eligible[args.train_tasks:count])}
        selection = "Hash-ranked tasks using local fixture servers; no reward-based selection"
    manifest = make_split(tasks, 0, args.seed)
    manifest.update(selected)
    if args.evaluate_training_tasks:
        manifest["validation"] = selected["train"]
    manifest["gym_revision"] = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    manifest["selection"] = selection
    if args.task_pool:
        manifest["selection"] += "; sampled from reward-selected task pool"
        manifest["task_pool"] = str(args.task_pool)
        manifest["task_pool_sha256"] = hashlib.sha256(args.task_pool.read_bytes()).hexdigest()
    manifest["excluded_tasks"] = args.exclude_task
    manifest["eligible_tasks"] = len(eligible)
    manifest["groups_per_task"] = args.groups_per_task
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "split.json").write_text(json.dumps(manifest, indent=2))
    frames = {}
    for split, names in selected.items():
        frames[split] = pd.DataFrame([{"data_source": f"toolathlon_gym/{split}", "agent_name": "toolathlon_decomposer",
                       "prompt": [{"role": "user", "content": f"Complete Gym task {name}."}],
                       "reward_model": {"style": "rule", "ground_truth": ""},
                       "extra_info": {"task_id": name, "split": split}, "index": i}
                      for i, name in enumerate(names)])
        output_frame = frames[split]
        if split == "train":
            output_frame = pd.concat([output_frame] * args.groups_per_task, ignore_index=True)
            output_frame["index"] = range(len(output_frame))
        output_frame.to_parquet(args.output / f"{split}.parquet")
    # A fixed training probe separates actual improvement from batch difficulty.
    # Held-out rows never enter the optimizer's train.parquet.
    probe = (frames["train"] if args.evaluate_training_tasks else frames["train"].head(4)).copy()
    probe["data_source"] = "toolathlon_gym/train_probe"
    pd.concat([probe] if args.evaluate_training_tasks else [probe, frames["validation"]], ignore_index=True).to_parquet(
        args.output / "evaluation.parquet")
    print(json.dumps({"eligible": len(eligible), **selected}, indent=2))


if __name__ == "__main__":
    main()
