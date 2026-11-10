"""Fixed training names, disjoint evaluation names, and a same-task train probe."""
import argparse
import json
from pathlib import Path

from gyms.synth.task import make_task


def prepare(directory, demo_validation=False, workplace_validation=False):
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
    validation = [row(t, "synth/train_probe") for t in train[:4]] + [row(t, "synth/heldout") for t in heldout]
    if demo_validation:
        from gyms.decomposer_demo.tasks import task_ids, task_spec
        for language in ('en', 'ru'):
            for task_id in task_ids():
                task = task_spec(task_id, language)
                validation.append({'data_source': f'decomposer-demo/{language}', 'agent_name': 'demo_decomposer',
                    'prompt': [{'role': 'user', 'content': task['prompt']}],
                    'reward_model': {'style': 'rule', 'ground_truth': task_id},
                    'extra_info': {'task_id': task_id, 'language': language}})
    if workplace_validation:
        from opd.workplace_assistant.prepare import validation_rows
        validation.extend(validation_rows())
    pd.DataFrame(validation).to_parquet(directory / "evaluation.parquet", index=False)
    (directory / "manifest.json").write_text(json.dumps({"version": "synth-swap-v1",
        "train_tasks": 16, "heldout_tasks": 8, "train_probe_tasks": 4,
        "attempts_per_task_per_epoch": 8, "demo_validation_tasks": 20 if demo_validation else 0,
        "demo_validation_rows": 40 if demo_validation else 0,
        "workplace_validation_tasks": 20 if workplace_validation else 0,
        "demo_validation_languages": ["en", "ru"] if demo_validation else []}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo-validation", action="store_true", help="Add held-out demo tasks in EN and RU")
    parser.add_argument("--workplace-validation", action="store_true")
    args = parser.parse_args()
    prepare(args.output, demo_validation=args.demo_validation, workplace_validation=args.workplace_validation)
