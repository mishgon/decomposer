"""Freeze three task pools from the complete baseline; prepare/check their datasets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess

from rl.toolathlon_gym.prepare_data import make_split

ROOT = Path(__file__).resolve().parent
PROFILES = ("full", "cold-start", "smoke")
GROUPS_PER_TASK = 3


def select_profiles(manifest, smoke_ids):
    if manifest["config"]["repetitions"] != 5:
        raise ValueError("Expected the completed five-attempt baseline")
    full, smoke = [], []
    for name in sorted(manifest["tasks"]):
        episodes = [manifest["episodes"][f"{name}/rep-{i:03d}"] for i in range(1, 6)]
        if any(e["status"] not in {"completed", "infrastructure_error"} for e in episodes):
            raise ValueError(f"Unfinished baseline task: {name}")
        scores = [float(e.get("partial_score") or 0) if e["status"] == "completed" else 0.
                  for e in episodes]
        row = {"task_id": name, "scores": scores,
               "mean_minutes": mean(e["elapsed_seconds"] for e in episodes) / 60,
               "score_range": max(scores) - min(scores)}
        if name in smoke_ids:
            smoke.append(row)
        # Native intermediate score is observed evidence, not inferred evaluator capability.
        if not any(0 < s < 1 for s in scores) or all(s >= .9 for s in scores):
            continue
        full.append(row)
    cold = [r for r in full if r["mean_minutes"] <= 30 and r["score_range"] > .1]
    if {r["task_id"] for r in smoke} != set(smoke_ids):
        raise ValueError("Smoke tasks must exist in the completed baseline")
    return dict(zip(PROFILES, (full, cold, smoke)))


def check_data(directory, profile):
    import pandas as pd
    pool_path = ROOT / "task_pools" / f"{profile}.json"
    pool = json.loads(pool_path.read_text())
    names = {r["task_id"] for r in pool["tasks"]}
    split = json.loads((directory / "split.json").read_text())
    if split.get("task_pool_sha256") != hashlib.sha256(pool_path.read_bytes()).hexdigest():
        raise ValueError("Dataset does not match the current frozen task pool")
    if set(split["train"]) != names or set(split["validation"]) != names:
        raise ValueError("Both train and evaluation must use the entire selected pool")
    groups = split.get("groups_per_task", GROUPS_PER_TASK)
    if type(groups) is not int or groups < 1:
        raise ValueError("groups_per_task must be a positive integer")
    for file, repeats in (("train", groups), ("evaluation", 1)):
        frame = pd.read_parquet(directory / f"{file}.parquet")
        counts = Counter(row["task_id"] for row in frame.extra_info)
        if counts != Counter({name: repeats for name in names}):
            raise ValueError(f"Incorrect {file} task/group counts")


def prepare_data(directory, profile, groups_per_task):
    """Use every task in one frozen pool; evaluation reuses the same tasks."""
    import pandas as pd
    if groups_per_task < 1:
        raise ValueError("groups_per_task must be positive")
    pool_path = ROOT / "task_pools" / f"{profile}.json"
    names = sorted(row["task_id"] for row in json.loads(pool_path.read_text())["tasks"])
    gym = ROOT.parents[1] / "external/toolathlon_gym"
    split = make_split(gym / "tasks/finalpool", 0, 42)
    if not names or len(names) != len(set(names)) or not set(names) <= set(split["train"]):
        raise ValueError("Task pool contains duplicate or unavailable tasks")
    split.update(train=names, validation=names, groups_per_task=groups_per_task,
                 gym_revision=subprocess.check_output(["git", "-C", str(gym), "rev-parse", "HEAD"], text=True).strip(),
                 selection="Entire frozen pool; evaluation reuses training tasks, no held-out validation",
                 task_pool=str(pool_path), task_pool_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest(),
                 excluded_tasks=[], eligible_tasks=len(names))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "split.json").write_text(json.dumps(split, indent=2))
    frame = pd.DataFrame([
        {"data_source": "toolathlon_gym/train", "agent_name": "toolathlon_decomposer",
         "prompt": [{"role": "user", "content": f"Complete Gym task {name}."}],
         "reward_model": {"style": "rule", "ground_truth": ""},
         "extra_info": {"task_id": name, "split": "train"}, "index": i}
        for i, name in enumerate(names)])
    training = pd.concat([frame] * groups_per_task, ignore_index=True)
    training["index"] = range(len(training))
    training.to_parquet(directory / "train.parquet")
    frame["data_source"] = "toolathlon_gym/train_probe"
    frame.to_parquet(directory / "evaluation.parquet")
    print(f"{len(names)} tasks; {groups_per_task} training groups/task; evaluation uses the same pool")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--manifest", type=Path, help="Regenerate frozen pools from baseline")
    action.add_argument("--prepare", type=Path, help="New dataset output directory")
    action.add_argument("--check-data", type=Path)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--groups-per-task", type=int, default=GROUPS_PER_TASK)
    args = parser.parse_args()
    if args.manifest:
        raw = args.manifest.read_bytes()
        smoke = json.loads((ROOT / "task_pools/smoke.json").read_text())
        pools = select_profiles(json.loads(raw), [r["task_id"] for r in smoke["tasks"]])
        (ROOT / "task_pools").mkdir(exist_ok=True)
        for name, tasks in pools.items():
            result = {"profile": name, "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                      "selection": "Observed 0<score<1; exclude five scores >=0.9; missing/infra=0. "
                                   "Cold-start additionally mean_minutes<=30 and range>0.1; smoke fixed two tasks.",
                      "tasks": tasks}
            if name == "smoke":
                result["selection"] = smoke["selection"]
            (ROOT / "task_pools" / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"{name}: {len(tasks)} tasks")
        return
    if not args.profile:
        parser.error("--profile is required")
    if args.prepare:
        prepare_data(args.prepare, args.profile, args.groups_per_task)
    check_data(args.prepare or args.check_data, args.profile)


if __name__ == "__main__":
    main()
