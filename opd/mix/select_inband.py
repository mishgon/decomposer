"""Freeze tasks with four resolved attempts from each in-band search pool."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from evals.wideseek.band import qualifies_band, resolved
from opd.wideseek.prepare import DEPTH_VALIDATION_PANEL, VALIDATION_PANEL


def select(runs, pools, output):
    if len(runs) != len(pools):
        raise ValueError('Supply one normalized task pool per run')
    cutoff = time.time()
    selected, evidence, provenance = {}, [], []
    initialization = None
    heldout = {t['question_sha256'] for panel in (VALIDATION_PANEL, DEPTH_VALIDATION_PANEL)
               for t in json.loads(panel.read_text())['heldout']}
    for run, pool in zip(runs, pools, strict=True):
        launch = json.loads((run.parent / 'launch.json').read_text())
        manifest = json.loads((run / 'manifest.json').read_text())
        settings = manifest['settings']
        raw = pool.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != settings['data_sha256']:
            raise ValueError(f'Task pool differs from the evaluated data: {pool}')
        if settings['repetitions'] != 4 or launch['max_pass_rate'] != .5:
            raise ValueError('Expected four attempts and a maximum pass rate of 0.5')
        current = launch['initialization']
        if current['tag'] != 'v1.7.0' or (initialization is not None and current != initialization):
            raise ValueError('All pools must use the same SFT v1.7.0 checkpoint')
        initialization = current
        tasks = {t['task_id']: t for line in raw.splitlines() if line.strip()
                 for t in [json.loads(line)]}
        mode, = settings['modes']
        qualified = 0
        for task_id in settings['tasks']:
            outcomes, attempts = [], []
            for number in range(1, 5):
                path = run / mode / task_id / f'attempt-{number:03d}' / 'result.json'
                if not path.exists():
                    continue
                result_raw = path.read_bytes()
                result = json.loads(result_raw)
                if result['task_id'] != task_id:
                    raise ValueError(f'Result task identity mismatch: {path}')
                if result.get('finished_at', float('inf')) > cutoff:
                    continue
                outcome = resolved(result, .9)
                if outcome is None:
                    continue
                outcomes.append(outcome)
                attempts.append({'path': str(path), 'sha256': hashlib.sha256(result_raw).hexdigest(),
                                 'passed': outcome, 'finished_at': result['finished_at']})
            if len(outcomes) != 4 or not qualifies_band(sum(outcomes), 4, .5, launch.get('require_success', False)):
                continue
            task = tasks[task_id]
            if task['question_sha256'] in heldout:
                raise ValueError(f'In-band task overlaps a frozen validation question: {task_id}')
            if task_id in selected and task != selected[task_id]:
                raise ValueError(f'Task differs between pools: {task_id}')
            selected[task_id] = task
            qualified += 1
            evidence.append({'task_id': task_id, 'run': str(run), 'passes': sum(outcomes),
                             'attempts': attempts})
        provenance.append({'run': str(run), 'pool': str(pool), 'pool_sha256': digest,
                           'require_success': launch.get('require_success', False), 'qualified': qualified})
    # Deduplicate identical questions even if they have different dataset IDs.
    unique = {}
    for task in sorted(selected.values(), key=lambda t: t['task_id']):
        unique.setdefault(task['question_sha256'], task)
    tasks = list(unique.values())
    if not tasks:
        raise ValueError('No qualifying tasks have four resolved attempts')
    raw = ''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks).encode()
    selection = {'cutoff': cutoff, 'initialization': initialization, 'runs': provenance,
                 'tasks': len(tasks), 'deduplicated_questions': len(selected) - len(tasks),
                 'task_ids': [t['task_id'] for t in tasks],
                 'sources': dict(Counter(t['source'] for t in tasks)),
                 'success_threshold': .9, 'max_pass_rate': .5, 'repetitions': 4,
                 'evidence': evidence, 'tasks_sha256': hashlib.sha256(raw).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'tasks.jsonl').write_bytes(raw)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--pools', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selection = select(args.runs, args.pools, args.output)
    print(json.dumps({k: v for k, v in selection.items() if k not in {'evidence', 'task_ids'}}), flush=True)
