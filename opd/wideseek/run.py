"""Validate one OPD experiment, record its configuration, then start veRL."""
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

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gyms.wideseek.runtime import save
from opd.wideseek.prepare import check
from opd.wideseek.preflight import check as preflight

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path.home() / 'models/decomposer-4b-sft')
    p.add_argument('--gpus', nargs=2, type=int, required=True, metavar=('TRAIN', 'ROLLOUT'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    args, overrides = p.parse_known_args()
    os.chdir(ROOT)
    if args.gpus[0] == args.gpus[1] or min(args.gpus) < 0:
        p.error('Choose two distinct GPU indices')
    model_path, data, output = args.model.resolve(), args.data.resolve(), args.output.resolve()
    if not (model_path / 'config.json').is_file():
        p.error('Model checkpoint is missing')
    data_manifest = check(data)
    for patch in (ROOT / 'opd/patches').glob('*.patch'):
        subprocess.run(['git', '-C', str(ROOT / 'external/verl'), 'apply', '--recount',
                        '--reverse', '--check', str(patch)], check=True, capture_output=True)
    env = {'OPD_ROOT': str(ROOT), 'OPD_DATA': str(data), 'OPD_ARTIFACTS': str(output),
           'MODEL_PATH': str(model_path), 'CUDA_VISIBLE_DEVICES': ','.join(map(str, args.gpus)),
           'PYTHONPATH': f'{ROOT}:{ROOT / "src"}:{ROOT / "external/verl"}',
           'TOKENIZERS_PARALLELISM': 'false', 'TENSORBOARD_DIR': str(output / 'tensorboard')}
    os.environ.update(env)
    os.environ.setdefault('WS_ARTIFACT_ROOT', str(ROOT / 'artifacts'))
    if not output.is_relative_to(Path(os.environ['WS_ARTIFACT_ROOT']).resolve()):
        p.error('Output must be within the worker WS_ARTIFACT_ROOT')
    # Never connect to another experiment's Ray cluster.
    os.environ['RAY_ADDRESS'] = 'local'
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    search = 'hydra.searchpath=[pkg://verl.trainer.config]'
    overrides = [f'trainer.experiment_name={output.name}', *overrides]
    with initialize_config_dir(config_dir=str(ROOT / 'opd/wideseek'), version_base=None):
        config = compose(config_name=args.config, overrides=[search, *overrides])
        resolved = OmegaConf.to_container(config, resolve=True)
    sources = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
               for folder in ('opd', 'gyms/wideseek', 'src/decomposer')
               for f in (ROOT / folder).rglob('*') if f.is_file() and f.suffix in ('.py', '.yaml', '.sh', '.patch', '.lock')}
    identity = {'config': resolved, 'data': data_manifest, 'sources': sources}
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (output.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / 'run.json'
        if args.resume:
            saved = json.loads(manifest.read_text())
            if saved['identity'] != identity:
                raise ValueError('Resume source, configuration or data differs')
            checkpoint_dir = Path(resolved['trainer']['default_local_dir'])
            if not (checkpoint_dir / 'latest_checkpointed_iteration.txt').exists():
                raise ValueError('No saved checkpoint to resume')
        else:
            output.mkdir(parents=True, exist_ok=False)
        state = {'identity': identity, 'pid': os.getpid(), 'started_at': time.time(),
                 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 'gpus': args.gpus, 'status': 'preflight', 'resume': args.resume}
        save(manifest, state)
        try:
            state['preflight'] = asyncio.run(preflight(model_path, output / 'preflight',
                worker_url=config.wideseek.worker_url,
                search_url=os.environ.get('WS_SEARCH_URL', 'http://127.0.0.1:18080')))
            if args.preflight_only:
                state['status'] = 'preflight_passed'
                return
            state['status'] = 'running'
            save(manifest, state)
            with (output / 'trainer.log').open('a') as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'opd.wideseek.trainer',
                    '--config-path', str(ROOT / 'opd/wideseek'), '--config-name', args.config,
                    search, *overrides], stdout=log, stderr=subprocess.STDOUT)
                old_handler = signal.signal(signal.SIGTERM, lambda *_: child.terminate())
                try:
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    child.wait()
                    raise
                finally:
                    signal.signal(signal.SIGTERM, old_handler)
            state.update(status='completed' if code == 0 else 'failed', exit_code=code)
            if code:
                raise SystemExit(code)
        except BaseException as exc:
            state.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            state['finished_at'] = time.time()
            save(manifest, state)


if __name__ == '__main__':
    main()
