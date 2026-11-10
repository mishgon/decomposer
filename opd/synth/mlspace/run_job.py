"""One container: verify infrastructure, evaluate stock weights, then smoke-test OPD."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from .experiments import collect_experiments
from opd.launch import gpu_overrides


HOSTS = ('lmrouter.2a2i.org', 'clearml.2a2i.org')


def configure_network():
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = ','.join(('localhost', '127.0.0.1', *HOSTS))
    for host in HOSTS:
        print(f'{host}: {socket.getaddrinfo(host, 443)[0][4][0]}', flush=True)


def preflight(output, model, key_file, config_path, *, workplace_validation=False):
    import httpx
    from clearml.utilities.pyhocon import ConfigFactory
    from transformers import AutoTokenizer
    from opd.teacher import score

    os.environ['LLM_PROXY_MASTER_KEY'] = key_file.read_text().strip()
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
    tokenizer = AutoTokenizer.from_pretrained(model)
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Copy file A to B.'},
                                        {'role': 'assistant', 'content': 'I will delegate the copy.'}],
                                       tokenize=True, return_dict=False, enable_thinking=False)
    asyncio.run(asyncio.wait_for(score(ids, tokenizer=tokenizer,
                                     output=output / 'teacher-preflight.json'), 90))
    if workplace_validation:
        from opd.workplace_assistant.preflight import check
        asyncio.run(check())
    print('ClearML authentication and teacher token-logprob alignment passed', flush=True)


def add_runtime_arguments(parser):
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lmrouter-key-file', type=Path, required=True)
    parser.add_argument('--clearml-config', type=Path, required=True)
    parser.add_argument('--num-gpus', type=int, required=True)
    parser.add_argument('--training-gpus', type=int, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--mode', choices=['baseline', 'train'], required=True)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    gpu_overrides(list(range(args.num_gpus)), args.training_gpus)
    experiment = next(e for e in collect_experiments() if e.name == args.experiment)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    os.environ['PATH'] = str(root / '.venv-opd/bin') + ':' + os.environ['PATH']
    configure_network()
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
    if not (args.model / 'config.json').exists():
        raise RuntimeError(f'Missing model: {args.model}')
    preflight(output, args.model, args.lmrouter_key_file, args.clearml_config,
              workplace_validation=args.mode == 'train')
    if args.mode == 'train':
        from opd.checkpoint_offload import check_access
        check_access(os.environ['OPD_CHECKPOINT_S3_URI'])
    baseline = output / 'baseline'
    manifest = baseline / 'manifest.json'
    if not (manifest.exists() and json.loads(manifest.read_text())['status'] == 'completed'):
        subprocess.run(['bash', 'evals/synth/baseline.sh', '0', str(args.model), str(baseline)], check=True)
    if args.mode == 'train':
        gpu_args = ['--gpus', *map(str, range(args.num_gpus)), '--training-gpus', str(args.training_gpus)]
        subprocess.run([sys.executable, '-u', '-m', 'opd.synth.run', *gpu_args,
                        '--baseline', str(baseline), '--model', str(args.model),
                        '--output', str(output / 'train'),
                        '--workplace-validation',
                        'actor_rollout_ref.rollout.val_kwargs.n=1',
                        'data.val_batch_size=32',
                        'trainer.max_actor_ckpt_to_keep=-1',
                        f'rollout.total_rollout_steps={experiment.rollout_steps}',
                        f'trainer.experiment_name={experiment.name}'], check=True)


if __name__ == '__main__':
    main()
