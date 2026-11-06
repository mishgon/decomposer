"""Compare hosted student and teacher on the frozen OPD panel with tight graph budgets."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

from decomposer.agent_server import agent_server
from decomposer.models import create_model
from evals.wideseek.run import summarize
from gyms.wideseek import REPO_ROOT, agents
from gyms.wideseek.run import episode
from gyms.wideseek.runtime import close_model, model_metadata, save


MODELS = {
    'student': 'lmrouter/qwen_3_5_4b_unlooped_non_thinking',
    'teacher': 'lmrouter/qwen_3_8_flash_next_non_thinking',
}
# Normal tool-loop paths, including middleware and LangGraph's termination check:
# lead: before_agent + 9*(model, after_model, tools) + model, after_model, after_agent.
# researcher: before_agent + 19*(before_model, model, after_model, tools) + final 3 nodes.
# Retry/jump paths can consume fewer graph steps per model call; these are graph caps.
LEAD_RECURSION_LIMIT = 32
RESEARCHER_RECURSION_LIMIT = 81


async def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    panel = json.loads((args.panel / 'manifest.json').read_text())
    raw = (args.panel / 'tasks.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != panel['files']['tasks.jsonl']:
        raise ValueError('Frozen panel task hash changed')
    tasks = [json.loads(line) for line in raw.splitlines()]
    if [t['task_id'] for t in tasks] != panel['train_task_ids'] + panel['heldout_task_ids']:
        raise ValueError('Frozen train/heldout panel does not match tasks')
    (root / 'tasks.jsonl').write_bytes(raw)
    os.environ['WS_ARTIFACT_ROOT'] = str(root)
    policies = {name: create_model(model) for name, model in MODELS.items()}
    manifest = {
        'status': 'running', 'started_at': time.time(), 'attempts': 48,
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
        'models': MODELS, 'generation': {k: model_metadata(v) for k, v in policies.items()},
        'researcher': agents.RESEARCHER_MODEL, 'judge': agents.JUDGE_MODEL,
        'recursion_limit': LEAD_RECURSION_LIMIT, 'agent_recursion_limit': RESEARCHER_RECURSION_LIMIT,
        'agent_run_budget_seconds': agents.AGENT_RUN_BUDGET_SECONDS,
        'agent_shutdown_grace_seconds': agents.AGENT_SHUTDOWN_GRACE_SECONDS,
        'episode_timeout': 2700, 'concurrency': args.concurrency,
        'panel': panel, 'episodes': [], 'errors': [],
    }
    save(root / 'manifest.json', manifest)
    rows = {name: [] for name in MODELS}
    semaphore = asyncio.Semaphore(args.concurrency)

    def update():
        summary = {}
        for name, results in rows.items():
            summary[name] = {}
            for split, ids in [('train', panel['train_task_ids']), ('heldout', panel['heldout_task_ids'])]:
                selected = [r for r in results if r['task_id'] in ids]
                summary[name][split] = summarize(selected)
        save(root / 'summary.json', summary)
        save(root / 'manifest.json', manifest)

    async def run(task, name, url):
        async with semaphore:
            settings = SimpleNamespace(agent='decomposer', worker_url=url, episode_timeout=2700,
                model_calls=None, output_tokens=None, recursion_limit=LEAD_RECURSION_LIMIT,
                agent_recursion_limit=RESEARCHER_RECURSION_LIMIT)
            try:
                result = await episode(task, 'decomposer', 1, root / name, settings, policy=policies[name])
                rows[name].append(result)
                manifest['episodes'].append({'model': name, 'task_id': task['task_id'],
                    'status': result['status'], 'evaluation': result['evaluation'],
                    'cleanup_errors': result.get('cleanup_errors', [])})
                print(f"{len(manifest['episodes'])}/48 {name} {task['task_id']} "
                      f"{result['status']} score={result['evaluation']['score']}", flush=True)
            except Exception as exc:
                manifest['errors'].append({'model': name, 'task_id': task['task_id'],
                                           'error': f'{type(exc).__name__}: {exc}'})
            update()

    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        async with agent_server(REPO_ROOT / 'gyms/wideseek/langgraph.json', port=port,
                                n_jobs_per_worker=64, python_executable=args.worker_python) as url:
            await asyncio.gather(*(run(task, name, url) for task in tasks for name in MODELS))
        manifest['status'] = 'completed_with_errors' if manifest['errors'] or any(
            r.get('cleanup_errors') or r['evaluation']['score'] is None
            for results in rows.values() for r in results) else 'completed'
    finally:
        if manifest['status'] == 'running':
            manifest['status'] = 'interrupted'
        manifest['finished_at'] = time.time()
        update()
        for policy in policies.values():
            await close_model(policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True, help='Frozen OPD data directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('Concurrency must be positive')
    asyncio.run(main(args))
