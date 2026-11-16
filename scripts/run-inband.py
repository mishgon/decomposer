"""Run student in-band evaluation against one local vLLM server."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import httpx

from evals.wideseek.band import compute
from evals.wideseek.clearml_reporter import collection_reporter
from gyms.wideseek import REPO_ROOT
from gyms.wideseek.run import cli, describe_run
from gyms.wideseek.runtime import save
from sft.wideseek.launch import create_parser, run_with_workers
from sft.wideseek.run import main as collect


async def collection(args):
    if args.import_run and not args.output.exists():
        _, root, manifest = await describe_run(args)
        previous = json.loads((args.import_run / 'manifest.json').read_text())
        # Relocation can change the host, health response, repetitions and concurrency.
        # Task bytes, models, sampling and episode budgets must remain identical.
        ignored = {'repetitions', 'concurrency', 'harness', 'retrieval'}
        for key, value in manifest['settings'].items():
            if key not in ignored and previous['settings'].get(key) != value:
                raise ValueError(f'Imported run differs in {key}')
        if previous['repetitions'] > args.n:
            raise ValueError('Cannot import more attempts than requested')
        root.mkdir(parents=True)
        count = 0
        for result in args.import_run.glob('decomposer/*/attempt-*/result.json'):
            shutil.copytree(result.parent, root / result.parent.relative_to(args.import_run))
            count += 1
        manifest['imported_run'] = {'path': str(args.import_run), 'revision': previous['revision'],
                                   'settings': previous['settings'], 'completed_attempts': count}
        save(root / 'manifest.json', manifest)
        print(f'Imported {count} completed attempts; retaining original raw traces', flush=True)
    args.resume = args.output.exists()
    await collect(args)


async def run(args):
    await run_with_workers(args, REPO_ROOT / 'gyms/wideseek/langgraph.json', collection)


def main():
    parser = create_parser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inference-python', type=Path, required=True)
    parser.add_argument('--import-run', type=Path)
    parser.add_argument('--max-pass-rate', type=float, default=.5)
    parser.add_argument('--clearml-project', default='decomposer-inband')
    parser.add_argument('--clearml-name', help='Defaults to the run directory name')
    args = parser.parse_args()
    if args.agent != 'student' or not 0 <= args.max_pass_rate <= 1:
        parser.error('Use --agent student and a pass-rate bound in [0, 1]')
    root = args.output.resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    initialization = json.loads((args.model / 'opd_initialization.json').read_text())
    if args.import_run:
        original = json.loads((args.import_run.parent / 'launch.json').read_text())
        if original['initialization'] != initialization:
            raise ValueError('Imported attempts used a different checkpoint')
    state = {'status': 'starting_inference', 'student_model': str(args.model),
             'initialization': initialization, 'repetitions': args.n, 'concurrency': args.concurrency,
             'max_pass_rate': args.max_pass_rate, 'started_at': time.time(),
             'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()}
    save(root / 'launch.json', state)
    command = [str(args.inference_python), '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(args.model), '--served-model-name', 'Qwen/Qwen3.5-4B',
        '--host', '127.0.0.1', '--port', '8024', '--max-model-len', '32768',
        '--max-num-seqs', str(args.concurrency), '--max-num-batched-tokens', '16384',
        '--gpu-memory-utilization', '0.80', '--language-model-only',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3',
        '--enable-prefix-caching', '--enforce-eager',
        '--additional-config', '{"gdn_prefill_backend":"triton"}']
    with collection_reporter(args), (root / 'vllm.log').open('a') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            with httpx.Client(trust_env=False, timeout=3) as client:
                for _ in range(600):
                    if server.poll() is not None:
                        raise RuntimeError('Inference exited; inspect vllm.log')
                    try:
                        if client.get('http://127.0.0.1:8024/health').is_success:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(2)
                else:
                    raise TimeoutError('Inference startup exceeded 20 minutes')
            state['status'] = 'running'
            save(root / 'launch.json', state)
            print('SFT inference ready; starting grouped in-band attempts', flush=True)
            cli(run, parser=parser)
            save(args.output / 'band.json', compute([args.output], args.success_threshold, args.max_pass_rate))
            state['status'] = 'completed'
        except BaseException as exc:
            state.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            state['finished_at'] = time.time()
            save(root / 'launch.json', state)


if __name__ == '__main__':
    main()
