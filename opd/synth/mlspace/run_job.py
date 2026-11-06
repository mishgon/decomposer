"""One container: verify infrastructure, evaluate stock weights, then smoke-test OPD."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from .experiments import HOME, MODEL, SANITY_ROOT, collect_experiments


HOSTS = ('lmrouter.2a2i.org', 'clearml.2a2i.org')


def configure_network():
    for key in list(os.environ):
        if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy'}:
            del os.environ[key]
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = ','.join(('localhost', '127.0.0.1', *HOSTS))
    for host in HOSTS:
        print(f'{host}: {socket.getaddrinfo(host, 443)[0][4][0]}', flush=True)


def preflight(output):
    import httpx
    from clearml.utilities.pyhocon import ConfigFactory
    from transformers import AutoTokenizer
    from opd.teacher import score

    os.environ['LLM_PROXY_MASTER_KEY'] = (HOME / '.config/decomposer/lmrouter.key').read_text().strip()
    config_path = HOME / '.config/clearml/matrosov-cds.conf'
    os.environ.update(CLEARML_CONFIG_FILE=str(config_path),
                      CLEARML_API_HOST='https://clearml.2a2i.org/api',
                      CLEARML_WEB_HOST='https://clearml.2a2i.org',
                      CLEARML_FILES_HOST='https://clearml.2a2i.org/files')
    config = ConfigFactory.parse_file(str(config_path))
    with httpx.Client(trust_env=False, timeout=20) as client:
        response = client.post(os.environ['CLEARML_API_HOST'] + '/auth.login',
                               auth=(config.get_string('api.credentials.access_key'),
                                     config.get_string('api.credentials.secret_key')),
                               json={'expiration_sec': 60})
        if response.status_code != 200 or not response.json().get('data', {}).get('token'):
            raise RuntimeError(f'ClearML authentication failed: HTTP {response.status_code}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Copy file A to B.'},
                                        {'role': 'assistant', 'content': 'I will delegate the copy.'}],
                                       tokenize=True, return_dict=False, enable_thinking=False)
    asyncio.run(asyncio.wait_for(score(ids, tokenizer=tokenizer,
                                     output=output / 'teacher-preflight.json'), 90))
    print('ClearML authentication and teacher token-logprob alignment passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--mode', choices=['baseline', 'train'], required=True)
    args = parser.parse_args()
    experiment = next(e for e in collect_experiments() if e.name == args.experiment)
    output = SANITY_ROOT / experiment.name
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    os.environ['PATH'] = str(root / '.venv-opd/bin') + ':' + os.environ['PATH']
    configure_network()
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
    if not (MODEL / 'config.json').exists():
        raise RuntimeError(f'Missing model: {MODEL}')
    preflight(output)
    if args.mode == 'train':
        from opd.checkpoint_offload import check_access
        check_access(os.environ['OPD_CHECKPOINT_S3_URI'])
    baseline = output / 'baseline'
    manifest = baseline / 'manifest.json'
    if not (manifest.exists() and json.loads(manifest.read_text())['status'] == 'completed'):
        subprocess.run(['bash', 'evals/synth/baseline.sh', '0', str(MODEL), str(baseline)], check=True)
    if args.mode == 'train':
        gpu_args = ['--gpu', '0'] if experiment.num_gpus == 1 else ['--gpus', '0', '1']
        subprocess.run([sys.executable, '-u', '-m', 'opd.synth.run', *gpu_args,
                        '--baseline', str(baseline), '--model', str(MODEL),
                        '--output', str(output / 'train'),
                        '--workplace-validation',
                        'actor_rollout_ref.rollout.val_kwargs.n=1',
                        'data.val_batch_size=32',
                        'trainer.max_actor_ckpt_to_keep=-1',
                        f'rollout.total_rollout_steps={experiment.rollout_steps}',
                        f'trainer.experiment_name={experiment.name}'], check=True)


if __name__ == '__main__':
    main()
