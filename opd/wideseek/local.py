"""Run the baseline and OPD with separate training and rollout GPUs on a Linux host."""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import sysconfig
import time

from gyms.wideseek import REPO_ROOT
from gyms.wideseek.runtime import save
from opd.launch import add_gpu_arguments, gpu_overrides


def check_inputs(args):
    from ray._common.utils import get_default_system_temp_dir
    from ray._private.utils import validate_socket_filepath

    if not (Path(sysconfig.get_path('include')) / 'Python.h').is_file():
        raise RuntimeError('Python development headers are required by Triton; use a uv-managed Python environment')
    # Match pinned Ray's session format, allowing any PID on this Linux host.
    max_pid = Path('/proc/sys/kernel/pid_max').read_text().strip()
    socket_path = (Path(get_default_system_temp_dir()).absolute() / 'ray' /
                   f'session_2000-01-01_00-00-00_000000_{max_pid}' / 'sockets/plasma_store')
    try:
        validate_socket_filepath(str(socket_path))
    except OSError as exc:
        raise ValueError('RAY_TMPDIR is too long for Ray Unix sockets; set a shorter shared-disk path') from exc
    if not (args.model / 'config.json').is_file():
        raise ValueError('Student checkpoint does not exist')
    if not args.source.is_file():
        raise ValueError('WideSeek source does not exist')
    if args.baseline:
        baseline = json.loads((args.baseline / 'manifest.json').read_text())
        if baseline['status'] != 'completed' or baseline['model'] != 'vllm/qwen_3_5_4b_non_thinking':
            raise ValueError('Reuse requires a completed coordinator baseline')
        if baseline.get('student_model') != str(args.model.resolve()):
            raise ValueError('Baseline must use the selected starting checkpoint')
        if baseline.get('source_sha256') != hashlib.sha256(args.source.read_bytes()).hexdigest():
            raise ValueError('Baseline and training WideSeek source differ')


@contextmanager
def service(command, log, health):
    import httpx
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            with httpx.Client(trust_env=False, timeout=5) as client:
                for _ in range(600):
                    if process.poll() is not None:
                        raise RuntimeError(f'Service exited; see {log}')
                    try:
                        if client.get(health).is_success:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(2)
                else:
                    raise TimeoutError(f'Service readiness timed out; see {log}')
            yield process
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def run(command):
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main(*, training_module='opd.wideseek.run'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Large-disk experiment directory; must be new')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--search-url', help='Reuse an existing retrieval service; leave its lifecycle to its owner')
    parser.add_argument('--baseline', type=Path, help='Reuse a completed starting-checkpoint baseline; training verifies its source hash')
    parser.add_argument('--resume-from', type=Path, help='Restore a complete veRL checkpoint into this new run')
    add_gpu_arguments(parser)
    args = parser.parse_args()
    gpu_overrides(args.gpus, args.training_gpus)
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    # Bare-metal Hertz-2 checkpoints stay on the shared disk.
    os.environ.pop('OPD_CHECKPOINT_S3_URI', None)
    os.chdir(REPO_ROOT)
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Review and commit changes before launching')
    check_inputs(args)
    # Only reserve ports for services this launch will start.
    ports = [] if args.search_url else [16333, 16334, 18080]
    for port in ports:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    if args.search_url:
        import httpx
        with httpx.Client(trust_env=False, timeout=30) as client:
            response = client.post(args.search_url.rstrip('/') + '/retrieve',
                                   json={'queries': ['Paris'], 'topk': 1})
            response.raise_for_status()
            if not response.json()['result'][0]:
                raise RuntimeError('Existing retrieval service returned no results')
    with ExitStack() as stack:
        args.root.parent.mkdir(parents=True, exist_ok=True)
        for gpu in sorted(args.gpus):
            lock = stack.enter_context((args.root.parent / f'.gpu-{gpu}.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            if used > 1024:
                raise RuntimeError(f'GPU {gpu} is occupied ({used} MiB)')
        args.root = args.root.resolve()
        args.root.mkdir(exist_ok=False)
        os.environ.update(WS_ASSETS=str(args.assets.resolve()), WS_TASK_SOURCE=str(args.source.resolve()),
                          WS_SEARCH_URL=args.search_url or 'http://127.0.0.1:18080', RAY_USE_UVLOOP='0')
        os.environ['PYTHONPATH'] = f"{REPO_ROOT / 'opd/wideseek/dns'}:{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}"
        os.environ['PATH'] = str(Path(sys.executable).parent) + ':' + os.environ['PATH']
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        state = {'status': 'running', 'pid': os.getpid(), 'started_at': time.time(),
                 'gpus': args.gpus, 'training_gpus': args.training_gpus, 'commit': commit, 'code_dir': str(REPO_ROOT),
                 'search_url': os.environ['WS_SEARCH_URL'], 'owns_retrieval': not bool(args.search_url)}
        save(args.root / 'local-run.json', state)
        try:
            if not args.search_url:
                stack.enter_context(service(['bash', 'gyms/wideseek/serve.sh', 'qdrant'],
                    args.root / 'qdrant.log', 'http://127.0.0.1:16333/healthz'))
                stack.enter_context(service(['bash', 'gyms/wideseek/serve.sh', 'retrieval'],
                    args.root / 'retrieval.log', 'http://127.0.0.1:18080/health'))
            if args.baseline:
                (args.root / 'baseline').symlink_to(args.baseline.resolve(), target_is_directory=True)
            else:
                run(['bash', 'opd/wideseek/baseline.sh', str(args.gpus[0]),
                                str(args.model.resolve()), str(args.root / 'baseline')])
            run([sys.executable, '-u', '-m', training_module,
                '--gpus', *map(str, args.gpus), '--training-gpus', str(args.training_gpus), '--source', str(args.source.resolve()),
                '--baseline', str(args.root / 'baseline'), '--model', str(args.model.resolve()),
                '--output', str(args.root / 'train'),
                *(['--resume-from', str(args.resume_from.resolve())] if args.resume_from else []),
                f'trainer.experiment_name={args.root.name}'])
            state['status'] = 'completed'
        except BaseException:
            state['status'] = 'failed'
            raise
        finally:
            state['finished_at'] = time.time()
            save(args.root / 'local-run.json', state)


if __name__ == '__main__':
    main()
