"""Evaluate a local SFT checkpoint on every teacher-solved WideSeek depth task."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from evals.wideseek.teacher_solved import prepare, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--teacher-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('-n', type=int, default=3)
    parser.add_argument('--concurrency', type=int, default=128)
    parser.add_argument('--episode-timeout', type=float, default=2700)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.n, args.concurrency, args.episode_timeout) < 1:
        parser.error('Counts and budgets must be positive')
    if not (args.model / 'config.json').is_file():
        parser.error('Expected a merged student checkpoint')

    from decomposer import models
    from gyms.wideseek import REPO_ROOT, agents
    from gyms.wideseek.run import cli
    from gyms.wideseek.runtime import save
    from opd.wideseek.local import service
    from sft.wideseek.launch import create_parser, main as collect

    root = args.output.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f'.gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if used > 1024:
            raise RuntimeError(f'GPU {args.gpu} is occupied ({used} MiB)')
        # Select a process-local endpoint for the existing Qwen model profile.
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            model_port = listener.getsockname()[1]
        models._QWEN_3_5_4B_PORT = model_port
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            worker_port = listener.getsockname()[1]
        state = {'student_model': str(args.model.resolve()), 'repetitions': args.n,
                 'concurrency': args.concurrency, 'episode_timeout': args.episode_timeout,
                 'max_model_len': 32768, 'gpu': args.gpu,
                 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()}
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO_ROOT, text=True).strip():
            raise RuntimeError('Review, commit and push changes before launching')
        if args.resume:
            previous = json.loads((root / 'evaluation.json').read_text())
            if any(previous[key] != value for key, value in state.items() if key != 'gpu'):
                raise ValueError('Resume requires the same checkpoint, code and evaluation settings')
            selection = json.loads((root / 'data/selection.json').read_text())
            if hashlib.sha256((root / 'data/tasks.jsonl').read_bytes()).hexdigest() != selection['tasks_sha256']:
                raise ValueError('Frozen task data changed')
        else:
            root.mkdir(exist_ok=False)
            selection = prepare(args.source, args.teacher_runs, root / 'data')
        initialization = args.model / 'opd_initialization.json'
        state.update(status='starting', started_at=time.time(), task_count=selection['task_count'],
                     checkpoint=json.loads(initialization.read_text()) if initialization.exists() else None)
        save(root / 'evaluation.json', state)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), VLLM_HOST='127.0.0.1',
                          WS_ARTIFACT_ROOT=str(root / 'eval'))
        agents.AGENT_MODELS['decomposer'] = 'vllm/qwen_3_5_4b_non_thinking'
        command = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', str(args.model.resolve()),
            '--served-model-name', 'Qwen/Qwen3.5-4B', '--host', '127.0.0.1', '--port', str(model_port),
            '--max-model-len', '32768', '--max-num-seqs', str(args.concurrency),
            '--gpu-memory-utilization', '0.80', '--language-model-only',
            '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3',
            '--enable-prefix-caching', '--enforce-eager', '--additional-config', '{"gdn_prefill_backend":"triton"}']
        argv = ['--data', str(root / 'data/tasks.jsonl'), '--output', str(root / 'eval'),
            '--agent', 'decomposer', '--limit', str(selection['task_count']), '-n', str(args.n),
            '--concurrency', str(args.concurrency), '--episode-timeout', str(args.episode_timeout),
            '--worker-port', str(worker_port), '--worker-slots', str(max(512, 4 * args.concurrency)),
            '--worker-python', str(args.worker_python), *(['--resume'] if args.resume else [])]
        try:
            with service(command, root / 'vllm.log', f'http://127.0.0.1:{model_port}/health'):
                state['status'] = 'running'
                save(root / 'evaluation.json', state)
                cli(collect, argv=argv, parser=create_parser())
            state['status'] = 'completed'
        except BaseException:
            state['status'] = 'interrupted'
            raise
        finally:
            save(root / 'task-success-counts.json', summarize(root / 'eval', selection['task_ids'], args.n))
            state['finished_at'] = time.time()
            save(root / 'evaluation.json', state)


if __name__ == '__main__':
    main()
