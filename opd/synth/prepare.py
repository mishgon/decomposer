"""Fixed training names, disjoint evaluation names, and a same-task train probe."""
import argparse
import json
from pathlib import Path

from gyms.synth.task import make_task


def prepare(directory):
    import pandas as pd
    train = [make_task("train", i) for i in range(16)]
    heldout = [make_task("eval", i) for i in range(8)]
    assert not ({name for t in train for name in t["initial"]} &
                {name for t in heldout for name in t["initial"]})
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in train + heldout))

    def row(task, source):
        return {"data_source": source, "agent_name": "synth_decomposer",
                "prompt": [{"role": "user", "content": task["prompt"]}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"task_id": task["task_id"]}}

    # Replication is explicit: 16 tasks x 8 attempts x 4 epochs = 512 episodes.
    pd.DataFrame([row(t, "synth/train") for t in train for _ in range(8)]).to_parquet(directory / "train.parquet", index=False)
    pd.DataFrame([row(t, "synth/train_probe") for t in train[:4]] +
                 [row(t, "synth/heldout") for t in heldout]).to_parquet(directory / "evaluation.parquet", index=False)
    (directory / "manifest.json").write_text(json.dumps({"version": "synth-swap-v1",
        "train_tasks": 16, "heldout_tasks": 8, "train_probe_tasks": 4,
        "attempts_per_task_per_epoch": 8, "validation_attempts": 4}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
