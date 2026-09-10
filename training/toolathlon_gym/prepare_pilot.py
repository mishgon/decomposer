"""Fixed, reward-blind pilot split drawn from tasks using local service fixtures."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from training.toolathlon_gym.prepare_data import make_split


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-tasks", type=int, default=16)
    parser.add_argument("--validation-tasks", type=int, default=8)
    args = parser.parse_args()
    root = Path("external/toolathlon_gym")
    tasks = root / "tasks/finalpool"
    local_servers = {"canvas", "notion", "google_sheet", "excel", "filesystem", "word",
                     "emails", "google_calendar", "woocommerce", "google_forms", "pptx"}
    eligible = [p.parent.name for p in tasks.glob("*/task_config.json")
                if set(json.loads(p.read_text())["needed_mcp_servers"]) <= local_servers]
    eligible.sort(key=lambda name: hashlib.sha256(f"{args.seed}:{name}".encode()).hexdigest())
    count = args.train_tasks + args.validation_tasks
    if min(args.train_tasks, args.validation_tasks) < 1 or count > len(eligible):
        raise ValueError("Invalid pilot split sizes")
    selected = {"train": sorted(eligible[:args.train_tasks]),
                "validation": sorted(eligible[args.train_tasks:count])}
    manifest = make_split(tasks, 0, args.seed)
    manifest.update(selected)
    manifest["gym_revision"] = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    manifest["selection"] = "Hash-ranked tasks using local fixture servers; no reward-based selection"
    manifest["eligible_tasks"] = len(eligible)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "split.json").write_text(json.dumps(manifest, indent=2))
    frames = {}
    for split, names in selected.items():
        frames[split] = pd.DataFrame([{"data_source": f"toolathlon_gym/{split}", "agent_name": "toolathlon_decomposer",
                       "prompt": [{"role": "user", "content": f"Complete Gym task {name}."}],
                       "reward_model": {"style": "rule", "ground_truth": ""},
                       "extra_info": {"task_id": name, "split": split}, "index": i}
                      for i, name in enumerate(names)])
        frames[split].to_parquet(args.output / f"{split}.parquet")
    # A fixed training probe separates actual improvement from batch difficulty.
    # Held-out rows never enter the optimizer's train.parquet.
    probe = frames["train"].head(4).copy()
    probe["data_source"] = "toolathlon_gym/train_probe"
    pd.concat([probe, frames["validation"]], ignore_index=True).to_parquet(
        args.output / "evaluation.parquet")
    print(json.dumps({"eligible": len(eligible), **selected}, indent=2))


if __name__ == "__main__":
    main()
