"""Run the agreed SFT 1.4.0 evaluation campaign from one server checkout."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx

from decomposer.models import create_model
from gyms.browsecomp import REPO_ROOT
from gyms.browsecomp.agents import AGENTS, JUDGE_MODEL
from gyms.browsecomp.run import preflight as retrieval_preflight, save
from gyms.widesearch.run import BACKEND

MODELS = ['decomposer_sft_1_4_0', 'researcher_4b_thinking', 'researcher_27b_thinking',
          'decomposer_4b_non_thinking', 'decomposer']


def jobs(data, gaia_tasks):
    tasks = sorted(json.loads(line)['task_id'] for line in data.read_text().split('\n') if line)
    if len(tasks) != 100 or len(set(tasks)) != 100:
        raise ValueError('Expected the fixed 100-task BrowseComp selection')
    result = []
    for phase in ['smoke', 'pilot', 'full']:
        for benchmark in ['browsecomp', 'widesearch', 'gaia2']:
            if phase == 'pilot' and benchmark != 'browsecomp':
                continue
            for agent in (MODELS if benchmark != 'gaia2' and phase != 'pilot' else [MODELS[0]]):
                for domain in (['execution', 'search'] if benchmark == 'gaia2' else [None]):
                    name = '/'.join(filter(None, [phase, benchmark, agent, domain]))
                    n = 1 if phase == 'smoke' else 3
                    concurrency = 1 if phase == 'smoke' or benchmark == 'widesearch' else 2
                    if benchmark == 'gaia2':
                        command = ['-m', 'evals.gaia2.run', '--agent', 'decomposer',
                            '--decomposer-model-id', AGENTS[MODELS[0]][0], '--agent-model-id', 'vllm/qwen_3_5_4b_thinking',
                            '--domains', domain, '--num-repeats', str(n), '--concurrency', str(concurrency),
                            '--agent-run-budget-seconds', '60', '--agent-shutdown-grace-seconds', '60', '--port-offset', '100']
                        if phase == 'smoke':
                            command += ['--task', gaia_tasks[domain]]
                    else:
                        command = ['-m', f'evals.{benchmark}.run' if benchmark == 'browsecomp' else 'gyms.widesearch.run',
                            '--agent', agent, '-n', str(n), '--concurrency', str(concurrency), '--timeout', '2700' if benchmark == 'browsecomp' else '7200',
                            '--retrieval-url', 'http://127.0.0.1:18082' if benchmark == 'browsecomp' else BACKEND]
                        if benchmark == 'browsecomp':
                            command += ['--data', str(data)]
                        command += ['--tasks', *(tasks[:5] if phase == 'pilot' else [tasks[0] if benchmark == 'browsecomp' else 'ws_en_001'])] if phase != 'full' else ['--all']
                    result.append({'name': name, 'benchmark': benchmark, 'agent': agent, 'domain': domain, 'command': command})
    return result


async def preflight(job):
    evidence = {}
    model_ids = [AGENTS[job['agent']][0], JUDGE_MODEL]
    if AGENTS[job['agent']][1] == 'decomposer':
        model_ids.append('vllm/qwen_3_5_4b_thinking')
    schema = {'type': 'function', 'function': {'name': 'search', 'description': 'Search',
              'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}}
    for model_id in dict.fromkeys(model_ids):
        model = create_model(model_id)
        try:
            async with asyncio.timeout(120):
                response = await model.bind_tools([schema], tool_choice='required').ainvoke('Search for Albert Einstein. Call search now.')
            evidence[model_id] = {'sampling': {k: getattr(model, k, None) for k in ['model_name', 'temperature', 'top_p', 'presence_penalty', 'reasoning_effort', 'extra_body']}, 'response': response.model_dump(mode='json')}
            if not response.tool_calls:
                raise RuntimeError(f'Model preflight returned no tool call: {model_id}')
        finally:
            model.http_client.close()
            await model.http_async_client.aclose()
    if job['benchmark'] == 'browsecomp':
        await retrieval_preflight('http://127.0.0.1:18082')
    elif job['benchmark'] == 'widesearch':
        async with httpx.AsyncClient(timeout=60, trust_env=False, headers={'X-API-Key': os.environ['KEENABLE_API_KEY']}) as client:
            response = await client.post('https://api.keenable.ai/v1/search', json={'query': 'Python official documentation', 'max_results': 1, 'mode': 'pro'})
            response.raise_for_status()
            evidence['search'] = response.json()
            hits = response.json()['results']
            if not hits:
                raise RuntimeError('Keenable preflight returned no results')
            response = await client.get('https://api.keenable.ai/v1/fetch', params={'url': hits[0]['url']})
            response.raise_for_status()
            evidence['fetch'] = response.json()
            if not response.json().get('content'):
                raise RuntimeError('Keenable preflight returned no page content')
    else:
        from gyms.gaia2.run import validate_preparation
        validate_preparation(domain=job['domain'])
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            response = await client.post(os.environ['LLM_PROXY_URL'].rstrip('/') + '/chat/completions',
                headers={'Authorization': 'Bearer ' + os.environ['LLM_PROXY_MASTER_KEY']},
                json={'model': 'nvidia/Llama-3.3-70B-Instruct-FP8', 'messages': [{'role': 'user', 'content': 'Reply OK'}], 'max_tokens': 8})
            response.raise_for_status()
    return evidence


def pilot_report(root, old_run):
    reports = {}
    pilot = next((root / 'pilot/browsecomp/decomposer_sft_1_4_0').glob('*/manifest.json'))
    tasks = set(json.loads(pilot.read_text())['tasks'])
    for name, path in [('old_60_60', old_run), ('new_180_180', pilot.parent)]:
        manifest = json.loads((path / 'manifest.json').read_text())
        episodes = [e for e in manifest['episodes'] if e['task'] in tasks]
        runs = []
        duplicates = 0
        for e in episodes:
            trace = json.loads((path / e['directory'] / 'trace.json').read_text())
            runs.extend(trace.get('agent_runs', {}).values())
            duplicates += bool(trace.get('shutdown_requested'))
        reports[name] = {'episodes': len(episodes), 'correct': sum(e['evaluation'].get('passed', False) for e in episodes),
            'worker_runs': len(runs), 'worker_timeouts': sum('TimeoutError' in str(r.get('error')) for r in runs),
            'worker_reports': sum(r.get('status') == 'responded' for r in runs), 'duplicate_shutdowns': duplicates,
            'missing_final_answer': sum(e['evaluation'].get('reason') == 'missing_final_answer' for e in episodes)}
    save(root / 'pilot-comparison.json', reports)


def run(args):
    from gyms.gaia2.run import dataset_manifest
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
        gaia_tasks = {}
        for domain in ['execution', 'search']:
            scenarios = json.loads(dataset_manifest(domain).read_text())['scenarios']
            if len(scenarios) != 160:
                raise ValueError(f'Expected 160 Gaia2 {domain} tasks')
            gaia_tasks[domain] = sorted(x['scenario_id'] for x in scenarios)[0]
        schedule = jobs(args.browsecomp_data.resolve(), gaia_tasks)
        state_file = root / 'queue.json'
        identity = {'revision': revision, 'data_sha256': hashlib.sha256(args.browsecomp_data.read_bytes()).hexdigest(), 'jobs': schedule}
        state = json.loads(state_file.read_text()) if state_file.exists() else {**identity, 'completed': []}
        if any(state.get(k) != v for k, v in identity.items()):
            raise ValueError('Queue code, data or schedule changed; use a new output directory')
        if state.get('status') == 'running' and state.get('child_pid'):
            try:
                os.kill(state['child_pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Previous child is still alive; do not start a second runner')
        (root / 'requirements-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        child = None
        try:
            for job in schedule:
                if job['name'] in state['completed']:
                    continue
                directory = root / job['name']
                directory.mkdir(parents=True, exist_ok=True)
                state.update(status='preflight', current=job['name'], child_pid=None, progress=None)
                state.pop('error', None)
                save(state_file, state)
                save(directory / 'preflight.json', asyncio.run(preflight(job)))
                manifests = list(directory.glob('*/manifest.json'))
                command = [sys.executable, *job['command'], '--output-dir', str(directory / 'results' if job['benchmark'] == 'gaia2' else directory)]
                if job['benchmark'] == 'gaia2' and (directory / 'results').exists():
                    if (directory / 'results' / job['domain'] / 'eval_metrics.json').exists():
                        command = None
                    else:
                        raise RuntimeError(f'Partial Gaia2 run preserved at {directory}; inspect before retrying')
                if manifests:
                    manifest = json.loads(manifests[0].read_text())
                    if manifest['status'] != 'completed':
                        command += ['--resume', str(manifests[0].parent)]
                    else:
                        command = None
                if command:
                    with (directory / 'runner.log').open('a') as log:
                        child = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=log, start_new_session=True)
                        state.update(status='running', child_pid=child.pid, command=command);save(state_file, state)
                        while child.poll() is None:
                            time.sleep(15)
                            found = list(directory.glob('*/manifest.json'))
                            if found:
                                m = json.loads(found[0].read_text())
                                state['progress'] = {'ended': len(m['episodes']), 'planned': len(m['tasks']) * m['repetitions'],
                                    'infrastructure_errors': sum(bool(e.get('infrastructure_errors')) for e in m['episodes']),
                                    'judge_errors': sum(e.get('evaluation', {}).get('status') == 'evaluation_error' for e in m['episodes'])}
                                save(state_file, state)
                        if child.returncode:
                            raise RuntimeError(f'{job["name"]} exited with {child.returncode}')
                if job['benchmark'] == 'gaia2':
                    metrics = json.loads((directory / 'results' / job['domain'] / 'eval_metrics.json').read_text())
                    errors = metrics['gym_metrics']['exception_types']
                    if any('Timeout' not in e and 'GraphRecursion' not in e for e in errors):
                        raise RuntimeError(f'Gaia2 infrastructure errors: {errors}')
                else:
                    manifest_path = next(directory.glob('*/manifest.json'));manifest = json.loads(manifest_path.read_text())
                    if manifest['status'] != 'completed':
                        raise RuntimeError(f'{job["name"]}: {manifest["status"]}')
                    if job['benchmark'] == 'widesearch':
                        with (directory / 'score.log').open('a') as log:
                            child = subprocess.Popen([sys.executable, '-m', 'evals.widesearch.run', str(manifest_path.parent)], cwd=REPO_ROOT, stdout=log, stderr=log, start_new_session=True)
                            state.update(status='running', child_pid=child.pid);save(state_file, state)
                            if child.wait() or json.loads((manifest_path.parent / 'evaluation/summary.json').read_text())['status'] != 'completed':
                                raise RuntimeError('WideSearch scoring needs attention')
                if job['name'].startswith('pilot/'):
                    pilot_report(root, args.old_browsecomp_run)
                state['completed'].append(job['name']);state.update(status='between_jobs', child_pid=None);save(state_file, state)
            state['status'] = 'completed'
        except BaseException as error:
            state.update(status='needs_attention', error=repr(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL);child.wait()
            save(state_file, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--browsecomp-data', type=Path, required=True)
    parser.add_argument('--old-browsecomp-run', type=Path, required=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    run(parser.parse_args())
