"""Freeze four mixed-gym tasks and two training episodes per task."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from gyms.wideseek.prepare import agent_input
from opd.wideseek.prepare import DEPTH_VALIDATION_PANEL, VALIDATION_PANEL

SOURCES = ('deepdive-qa_rl', 'deepdive-qa_sft', 'depth', 'width')


def prepare(source, output):
    import pandas as pd

    raw = source.read_bytes()
    excluded = {entry['question_sha256'] for panel in (VALIDATION_PANEL, DEPTH_VALIDATION_PANEL)
                for entry in json.loads(panel.read_text())['heldout']}
    pools = {name: {} for name in SOURCES}
    for line in raw.splitlines():
        task = json.loads(line)
        if task['source'] in pools and task['question_sha256'] not in excluded:
            pools[task['source']].setdefault(task['question_sha256'], task)
    rng = random.Random(42)
    selected = []
    for name, pool in pools.items():
        candidates = sorted(pool.values(), key=lambda t: t['task_id'])
        candidates = [t for t in candidates if t['question_sha256'] not in excluded]
        if not candidates:
            raise ValueError(f'Smoke requires an eligible task from {name}')
        task = rng.choice(candidates)
        selected.append(task)
        excluded.add(task['question_sha256'])
    if len({t['task_id'] for t in selected}) != 4:
        raise ValueError('Smoke task IDs must be unique')

    def row(task):
        gym = 'deepdive' if task['source'].startswith('deepdive-') else 'wideseek'
        return {'data_source': f'{gym}/train', 'agent_name': 'wideseek_decomposer',
                'prompt': agent_input(task)['messages'],
                'reward_model': {'style': 'rule', 'ground_truth': ''},
                'extra_info': {'task_id': task['task_id']}}

    output.mkdir(parents=True, exist_ok=False)
    (output / 'tasks.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in selected))
    rows = [row(t) for t in selected]
    # One epoch visits all eight rows once. Rollout n stays at its production value of one.
    pd.DataFrame(rows * 2).to_parquet(output / 'train.parquet', index=False)
    # veRL constructs a validation loader even when validation is disabled.
    pd.DataFrame(rows).to_parquet(output / 'evaluation.parquet', index=False)
    manifest = {'smoke': True, 'seed': 42, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'task_ids': [t['task_id'] for t in selected],
                'train_task_ids': [t['task_id'] for t in selected], 'train_tasks': 4,
                'heldout_task_ids': [], 'heldout_tasks': 0, 'train_probe_tasks': 0,
                'workplace_validation_tasks': 0, 'attempts_per_task_per_epoch': 2,
                'training_sources': dict(Counter(t['source'] for t in selected)),
                'evaluation': 'Disabled for smoke; evaluation.parquet is a loader placeholder',
                'files': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ('tasks.jsonl', 'train.parquet', 'evaluation.parquet')}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
