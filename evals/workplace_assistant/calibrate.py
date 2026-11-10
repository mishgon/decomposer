"""Measure a candidate panel with hosted coordinators and a fixed Qwen4B worker."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from decomposer.models import create_model
from gyms.workplace_assistant.in_process import episode


async def main(args):
    rows = json.loads(args.tasks.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    policy = create_model(args.model)
    worker = create_model('lmrouter/qwen_3_5_4b_non_thinking')
    manifest = {'benchmark': 'workplace-assistant-calibration', 'status': 'running',
                'model': args.model, 'worker_model': 'lmrouter/qwen_3_5_4b_non_thinking',
                'tasks': len(rows), 'repetitions': args.repetitions, 'concurrency': args.concurrency,
                'started_at': time.time(), 'episodes': []}
    def save():
        temp = args.output / 'manifest.tmp'
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(args.output / 'manifest.json')
    save()
    semaphore = asyncio.Semaphore(args.concurrency)
    async def run(row, attempt):
        async with semaphore:
            result = await episode(row, policy, worker, timeout=args.timeout)
            result['attempt'] = attempt
            (args.output / f'{row["id"]}-{attempt}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
            manifest['episodes'].append({k: result[k] for k in ('task_id', 'attempt', 'environment',
                'status', 'error_type', 'score', 'passed', 'native_correct', 'elapsed_seconds')})
            save()
            print(f'{len(manifest["episodes"])}/{len(rows)*args.repetitions}: {row["id"]} score={result["score"]}', flush=True)
    try:
        async with asyncio.TaskGroup() as group:
            for row in rows:
                for attempt in range(1, args.repetitions + 1):
                    group.create_task(run(row, attempt))
        manifest['status'] = 'completed'
    finally:
        if manifest['status'] == 'running':
            manifest['status'] = 'interrupted'
        manifest['finished_at'] = time.time()
        save()
        for model in (policy, worker):
            model.http_client.close()
            await model.http_async_client.aclose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=300)
    asyncio.run(main(parser.parse_args()))
