"""Freeze selected Gym tasks and write veRL inputs without exposing references."""
import argparse
import hashlib
import json
from pathlib import Path

from gyms.wideseek.prepare import agent_input


def prepare(source, output, task_ids):
    import pandas as pd
    raw = source.read_bytes()
    all_tasks = {t['task_id']: t for t in map(json.loads, raw.splitlines())}
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError('Choose distinct task IDs')
    selected = [all_tasks[t] for t in task_ids]
    if len({t['question_sha256'] for t in selected}) != len(selected):
        raise ValueError('Selected tasks contain duplicate questions')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'tasks.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in selected))
    rows = [{'data_source': 'wideseek/train', 'agent_name': 'wideseek_decomposer',
             'prompt': agent_input(t)['messages'], 'reward_model': {'style': 'rule', 'ground_truth': ''},
             'extra_info': {'task_id': t['task_id']}, 'index': i} for i, t in enumerate(selected)]
    pd.DataFrame(rows).to_parquet(output / 'train.parquet', index=False)
    for r in rows:
        r['data_source'] = 'wideseek/train_probe'
    pd.DataFrame(rows).to_parquet(output / 'evaluation.parquet', index=False)
    files = ('tasks.jsonl', 'train.parquet', 'evaluation.parquet')
    manifest = {'task_ids': task_ids, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'evaluation': 'Same tasks as training; an overfit probe, no held-out validation',
                'files': {f: hashlib.sha256((output / f).read_bytes()).hexdigest() for f in files}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def check(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Prepared data changed: {name}')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=['width-00001', 'width-00003'])
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output, a.tasks), indent=2))
