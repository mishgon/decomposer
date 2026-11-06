"""Starting-checkpoint coordinator baseline: eight held-out WideSeek tasks, four attempts each."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace

from decomposer.agent_server import agent_server
from decomposer.models import create_model
from gyms.wideseek import REPO_ROOT
from gyms.wideseek.run import episode
from gyms.wideseek.runtime import close_model, save
from opd.wideseek.prepare import prepare


async def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    panel = prepare(args.source, root / 'data')
    tasks = [json.loads(line) for line in (root / 'data/tasks.jsonl').read_text().splitlines()]
    tasks = [t for t in tasks if t['task_id'] in panel['heldout_task_ids']]
    os.environ['WS_ARTIFACT_ROOT'] = str(root)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    manifest = {'status': 'running', 'model': 'vllm/qwen_3_5_4b_non_thinking',
                'student_model': str(args.model.resolve()),
                'episode_timeout': 900,
                'tasks': 8, 'repetitions': 4, 'concurrency': 8, 'episodes': [],
                'source_sha256': panel['source_sha256'], 'started_at': time.time()}
    save(root / 'manifest.json', manifest)
    policy = create_model(manifest['model'])
    semaphore = asyncio.Semaphore(8)

    async def run(task, attempt, url):
        async with semaphore:
            settings = SimpleNamespace(agent='decomposer', worker_url=url, episode_timeout=900,
                                       model_calls=None, output_tokens=None)
            result = await episode(task, 'decomposer', attempt, root, settings, policy=policy)
            if result.get('cleanup_errors'):
                raise RuntimeError(f'Baseline worker cleanup failed: {task["task_id"]}')
            score = result['evaluation']['score']
            if score is None and result['status'] == 'timeout':
                score = 0.0
            if score is None:
                raise RuntimeError(f'Unscored baseline episode: {task["task_id"]}; inspect saved result')
            manifest['episodes'].append({'task_id': task['task_id'], 'attempt': attempt,
                'status': result['status'], 'score': score,
                'passed': result['status'] == 'finished' and score == 1})
            save(root / 'manifest.json', manifest)
    try:
        async with agent_server(REPO_ROOT / 'gyms/wideseek/langgraph.json', port=port,
                                n_jobs_per_worker=64, python_executable=args.worker_python) as url:
            async with asyncio.TaskGroup() as group:
                for task in tasks:
                    for attempt in range(1, 5):
                        group.create_task(run(task, attempt, url))
        rows = manifest['episodes']
        save(root / 'summary.json', {'attempts': len(rows), 'tasks': 8,
             'mean_score': sum(r['score'] for r in rows) / len(rows),
             'pass@1': sum(r['passed'] for r in rows) / len(rows),
             'errors': sum(r['status'] != 'finished' for r in rows)})
        manifest['status'] = 'completed'
    finally:
        if manifest['status'] == 'running':
            manifest['status'] = 'interrupted'
        manifest['finished_at'] = time.time()
        save(root / 'manifest.json', manifest)
        await close_model(policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-python', type=Path, default=REPO_ROOT / '.venv-workers/bin/python')
    asyncio.run(main(parser.parse_args()))
