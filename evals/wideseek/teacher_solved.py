"""Freeze teacher-solved depth tasks and summarize repeated student evaluations."""
from collections import Counter
import hashlib
import json
import random


def prepare(source, teacher_runs, output):
    raw = source.read_bytes()
    tasks = [json.loads(line) for line in raw.splitlines()]
    by_id = {task['task_id']: task for task in tasks}
    if len(by_id) != len(tasks) or any(not key.startswith('depth-') for key in by_id):
        raise ValueError('Expected unique WideSeek depth tasks')
    evidence = {}
    for root in teacher_runs:
        for path in sorted(root.glob('decomposer/depth-*/attempt-*/result.json')):
            result = json.loads(path.read_text())
            evaluation = result.get('evaluation', {})
            if evaluation.get('status') != 'scored' or evaluation.get('score') != 1:
                continue
            if evaluation.get('metric') != 'qa_accuracy':
                raise ValueError(f'Expected depth QA accuracy: {path}')
            trace_path = (root / result['trace_path'] if result.get('trace_path') else
                          path.parent / result['execution_directory'] / 'trace.json')
            trace = json.loads(trace_path.read_text())
            if 'flash' not in trace.get('model', '').lower():
                continue
            names = {call.get('name', call.get('function', {}).get('name'))
                     for message in trace['messages']
                     for call in message.get('data', message).get('tool_calls', [])}
            if 'spawn_subagent' in names:
                continue
            task_id = result['task_id']
            if task_id not in by_id:
                raise ValueError(f'Teacher task missing from source: {task_id}')
            evidence.setdefault(task_id, {'result': str(path.resolve()),
                'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'trace': str(trace_path.resolve()), 'teacher': trace['model']})
    if not evidence:
        raise ValueError('No teacher-solved depth tasks found')
    ids = sorted(evidence)
    random.Random(42).shuffle(ids)
    output.mkdir(parents=True, exist_ok=False)
    selected = ''.join(json.dumps(by_id[key], ensure_ascii=False) + '\n' for key in ids).encode()
    (output / 'tasks.jsonl').write_bytes(selected)
    manifest = {'rule': 'Flash Next coordinator, new tool era, depth QA accuracy = 1',
                'seed': 42, 'task_ids': ids, 'task_count': len(ids),
                'source': str(source.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'tasks_sha256': hashlib.sha256(selected).hexdigest(), 'evidence': evidence}
    (output / 'selection.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def summarize(root, task_ids, repetitions):
    rows = []
    for task_id in task_ids:
        scores = []
        unresolved = []
        for attempt in range(1, repetitions + 1):
            path = root / 'decomposer' / task_id / f'attempt-{attempt:03d}' / 'result.json'
            result = json.loads(path.read_text()) if path.exists() else {}
            evaluation = result.get('evaluation', {})
            if evaluation.get('status') == 'scored' and evaluation.get('score') in (0, 1):
                scores.append(evaluation['score'])
            elif result.get('status') == 'timeout':
                scores.append(0)
            else:
                unresolved.append(attempt)
        rows.append({'task_id': task_id, 'successes': int(sum(scores)),
                     'resolved_attempts': len(scores), 'unresolved_attempts': unresolved,
                     'bucket': int(sum(scores)) if not unresolved else None})
    complete = [row for row in rows if row['bucket'] is not None]
    return {'repetitions': repetitions, 'tasks': rows, 'complete_tasks': len(complete),
            'success_count_buckets': dict(Counter(row['bucket'] for row in complete)),
            'pass_at_n': sum(row['successes'] > 0 for row in complete) / len(complete) if complete else None}
