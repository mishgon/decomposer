"""Finish the transfer panel with bounded concurrency and an early stop."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time

from decomposer.models import create_model
from gyms.workplace_assistant.in_process import episode
from evals.workplace_assistant.select_transfer import candidates, results, select


async def main(args):
    student = results(args.root, ['student', 'student-remaining'])
    teacher = results(args.root, ['teacher', 'teacher-transfer'])
    pool = candidates(json.loads((args.root / 'full.json').read_text()), student)
    remaining = [r for r in pool if r['id'] not in teacher]
    queues = {passed: [r for r in remaining if student[r['id']]['passed'] == passed]
              for passed in (False, True)}
    args.output.mkdir(exist_ok=False)
    manifest = {'benchmark': 'workplace-assistant-calibration', 'status': 'running',
                'model': 'lmrouter/qwen_3_8_flash_next_non_thinking',
                'worker_model': 'lmrouter/qwen_3_5_4b_unlooped_non_thinking',
                'tasks': len(remaining), 'repetitions': 1, 'concurrency': 8,
                'started_at': time.time(), 'episodes': [], 'reused_teacher_tasks': len(teacher)}
    def save():
        path = args.output / 'manifest.tmp'
        path.write_text(json.dumps(manifest, indent=2))
        path.replace(args.output / 'manifest.json')
    policy = create_model(manifest['model'])
    worker = create_model(manifest['worker_model'])
    pending = {}
    save()
    try:
        while True:
            chosen, progress = select(pool, student, teacher)
            manifest['selection_progress'] = progress
            save()
            if chosen:
                manifest['stop_reason'] = '20-task target reached'
                break
            # Reserve slots for controls until two shared passes are measured.
            while len(pending) < 8:
                controls_pending = sum(student[r['id']]['passed'] for r in pending.values())
                need_controls = progress['both_pass'] < 2
                need_gap = progress['teacher_only_passes'] < 18
                if need_controls and queues[True] and (controls_pending < 2 or not need_gap):
                    row = queues[True].pop(0)
                elif need_gap and queues[False]:
                    row = queues[False].pop(0)
                else:
                    break
                pending[asyncio.create_task(episode(row, policy, worker))] = row
            if not pending:
                manifest['stop_reason'] = 'candidate pool exhausted before target'
                break
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                row = pending.pop(task)
                result = task.result()
                result['attempt'] = 1
                (args.output / f'{row["id"]}-1.json').write_text(json.dumps(result, indent=2))
                summary = {k: result[k] for k in ('task_id', 'attempt', 'environment', 'status',
                    'error_type', 'score', 'passed', 'native_correct', 'elapsed_seconds')}
                teacher[row['id']] = summary
                manifest['episodes'].append(summary)
                save()
        manifest['status'] = 'completed'
    finally:
        manifest['cancelled_task_ids'] = [r['id'] for r in pending.values()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if manifest['status'] == 'running':
            manifest['status'] = 'interrupted'
        manifest['finished_at'] = time.time()
        save()
        for model in (policy, worker):
            model.http_client.close()
            await model.http_async_client.aclose()
    subprocess.run([sys.executable, '-m', 'evals.workplace_assistant.select_transfer',
        '--root', str(args.root), '--output', str(args.root / 'transfer-selection'),
        '--teacher-labels', 'teacher', 'teacher-transfer', args.output.name], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
