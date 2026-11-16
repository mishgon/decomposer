"""Freeze tasks outside the SFT task index, existing pool, and OPD holdouts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from opd.wideseek.prepare import DEPTH_VALIDATION_PANEL, VALIDATION_PANEL


def prepare(sources, exclude_pool, sft_index, output):
    tasks = {}
    for source in sources:
        for line in source.read_text().splitlines():
            task = json.loads(line)
            if task['task_id'] in tasks:
                raise ValueError(f"Duplicate source task ID: {task['task_id']}")
            tasks[task['task_id']] = task
    index = json.loads(sft_index.read_text())
    sft_ids = set(index['task_ids'])
    if missing := sft_ids - tasks.keys():
        raise ValueError(f'SFT task IDs missing from source datasets: {sorted(missing)[:10]}')
    excluded = [json.loads(line) for line in exclude_pool.read_text().splitlines()]
    excluded_ids = sft_ids | {t['task_id'] for t in excluded}
    excluded_hashes = {tasks[t]['question_sha256'] for t in sft_ids}
    excluded_hashes.update(t['question_sha256'] for t in excluded)
    excluded_hashes.update(t['question_sha256'] for panel in (VALIDATION_PANEL, DEPTH_VALIDATION_PANEL)
                           for t in json.loads(panel.read_text())['heldout'])
    selected, seen = [], set()
    for task in sorted(tasks.values(), key=lambda t: t['task_id']):
        question = task['question_sha256']
        if task['task_id'] in excluded_ids or question in excluded_hashes or question in seen:
            continue
        selected.append(task)
        seen.add(question)
    if not selected:
        raise ValueError('No unseen tasks remain')
    random.Random(42).shuffle(selected)
    output.mkdir(parents=True, exist_ok=False)
    data = output / 'tasks.jsonl'
    data.write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in selected))
    manifest = {'seed': 42, 'tasks': len(selected), 'counts': dict(Counter(t['source'] for t in selected)),
                'source_tasks': len(tasks), 'sft_task_ids': len(sft_ids), 'sft_provenance': index,
                'excluded_pool_tasks': len(excluded), 'historical_success_required': False,
                'deduplication': 'question_sha256',
                'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [*sources, exclude_pool, sft_index]},
                'tasks_sha256': hashlib.sha256(data.read_bytes()).hexdigest()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--exclude-pool', type=Path, required=True)
    parser.add_argument('--sft-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.sources, args.exclude_pool, args.sft_index, args.output)
    print(json.dumps({k: result[k] for k in ('tasks', 'counts', 'source_tasks', 'sft_task_ids', 'tasks_sha256')}))
