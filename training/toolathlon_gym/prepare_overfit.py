"""One task for a four-rollout diagnostic, deliberately without a holdout."""

import argparse
import json
from pathlib import Path


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = Path("external/toolathlon_gym/tasks/finalpool").resolve()
    task = (tasks / args.task).resolve()
    if task.parent != tasks or not (task / "task_config.json").is_file():
        parser.error("Unknown Gym task")
    args.output.mkdir(parents=True, exist_ok=False)
    row = {"data_source": "toolathlon_gym/overfit_probe", "agent_name": "toolathlon_decomposer",
           "prompt": [{"role": "user", "content": f"Complete Gym task {args.task}."}],
           "reward_model": {"style": "rule", "ground_truth": ""}, "index": 0,
           "extra_info": {"task_id": args.task, "split": "overfit"}}
    for name in ("train", "evaluation"):
        pd.DataFrame([row]).to_parquet(args.output / f"{name}.parquet")
    (args.output / "split.json").write_text(json.dumps({
        "train": [args.task], "validation": [], "overfit_probe": [args.task],
        "selection": "Diagnostic task selected for runtime and variable reward; NOT held out",
    }, indent=2))


if __name__ == "__main__":
    main()
