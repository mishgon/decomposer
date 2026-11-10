"""Freeze selected Gym tasks and write veRL inputs without exposing references."""
import argparse
import hashlib
import json
from pathlib import Path

from gyms.wideseek.prepare import agent_input

VALIDATION_PANEL = Path(__file__).with_name("validation_panel.json")
DEPTH_VALIDATION_PANEL = Path(__file__).with_name("validation_panel_depth.json")


def prepare(source, output):
    import pandas as pd
    raw = source.read_bytes()
    # Preserve dataset order, dropping duplicate question hashes before splitting.
    unique = {}
    for line in raw.splitlines():
        task = json.loads(line)
        unique.setdefault(task['question_sha256'], task)
    selected = list(unique.values())
    by_id = {t['task_id']: t for t in selected}
    if len(by_id) != len(selected):
        raise ValueError('WideSeek task IDs must be unique')
    sources = {t['source'] for t in selected}
    if sources not in ({'width'}, {'depth'}):
        raise ValueError('OPD requires one WideSeek split: width or depth')
    validation_panel = DEPTH_VALIDATION_PANEL if sources == {'depth'} else VALIDATION_PANEL
    frozen = json.loads(validation_panel.read_text())
    for entry in frozen['heldout'] + frozen['train_probe']:
        task = by_id.get(entry['task_id'])
        if task is None or task['question_sha256'] != entry['question_sha256']:
            raise ValueError(f"Frozen validation task changed or missing: {entry['task_id']}")
    heldout = [by_id[t['task_id']] for t in frozen['heldout']]
    heldout_hashes = {t['question_sha256'] for t in heldout}
    train = [t for t in selected if t['question_sha256'] not in heldout_hashes]
    probes = [by_id[t['task_id']] for t in frozen['train_probe']]
    panel_path = Path(__file__).resolve().parents[2] / 'gyms/workplace_assistant/validation20.json'
    from opd.workplace_assistant.prepare import validation_rows
    workplace = validation_rows()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'tasks.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in selected))

    def row(task, source):
        return {'data_source': source, 'agent_name': 'wideseek_decomposer',
                'prompt': agent_input(task)['messages'],
                'reward_model': {'style': 'rule', 'ground_truth': ''},
                'extra_info': {'task_id': task['task_id']}}

    pd.DataFrame([row(t, 'wideseek/train') for t in train]).to_parquet(
        output / 'train.parquet', index=False)
    validation = [row(t, 'wideseek/train_probe') for t in probes]
    validation += [row(t, 'wideseek/heldout') for t in heldout]
    validation += workplace
    pd.DataFrame(validation).to_parquet(output / 'evaluation.parquet', index=False)
    files = ('tasks.jsonl', 'train.parquet', 'evaluation.parquet')
    manifest = {'task_ids': [t['task_id'] for t in selected],
                'train_task_ids': [t['task_id'] for t in train],
                'heldout_task_ids': [t['task_id'] for t in heldout],
                'train_tasks': len(train), 'heldout_tasks': len(heldout), 'train_probe_tasks': len(probes),
                'attempts_per_task_per_epoch': 1, 'workplace_validation_tasks': len(workplace),
                'source': next(iter(sources)),
                'validation_panel_sha256': hashlib.sha256(validation_panel.read_bytes()).hexdigest(),
                'source_sha256': hashlib.sha256(raw).hexdigest(),
                'workplace_panel_sha256': hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                'evaluation': '4 train probes, 8 disjoint WideSeek questions, frozen Workplace-20; pass@1',
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
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output), indent=2))
