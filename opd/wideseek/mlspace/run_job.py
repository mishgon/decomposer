"""One container: verify infrastructure, evaluate stock weights, then run WideSeek OPD."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .experiments import collect_experiments
from opd.launch import gpu_overrides
from opd.synth.mlspace.run_job import add_runtime_arguments, configure_network, preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--mode', choices=['baseline', 'train'], required=True)
    add_runtime_arguments(parser)
    parser.add_argument('--source', type=Path, required=True)
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
    import httpx
    with httpx.Client(trust_env=False, timeout=30) as client:
        response = client.post(os.environ["WS_SEARCH_URL"].rstrip("/") + "/retrieve",
                               json={"queries": ["Paris"], "topk": 1})
        response.raise_for_status()
        if not response.json()["result"][0]:
            raise RuntimeError("WideSeek retrieval returned no passages")
    if args.mode == 'train':
        from opd.checkpoint_offload import check_access
        check_access(os.environ['OPD_CHECKPOINT_S3_URI'])
    baseline = output / 'baseline'
    manifest = baseline / 'manifest.json'
    if not (manifest.exists() and json.loads(manifest.read_text())['status'] == 'completed'):
        subprocess.run(['bash', 'opd/wideseek/baseline.sh', '0', str(args.model), str(baseline)], check=True)
    if args.mode == 'train':
        gpu_args = ['--gpus', *map(str, range(args.num_gpus)), '--training-gpus', str(args.training_gpus)]
        subprocess.run([sys.executable, '-u', '-m', 'opd.wideseek.run', *gpu_args,
                        '--baseline', str(baseline), '--model', str(args.model),
                        '--output', str(output / 'train'),
                        '--source', str(args.source),
                        'actor_rollout_ref.rollout.val_kwargs.n=1',
                        'data.val_batch_size=32',
                        'trainer.max_actor_ckpt_to_keep=-1',
                        f'rollout.total_rollout_steps={experiment.rollout_steps}',
                        f'trainer.experiment_name={experiment.name}'], check=True)


if __name__ == '__main__':
    main()
