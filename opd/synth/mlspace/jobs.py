"""Submit committed synthetic experiments using the installed MLSpace library."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from .experiments import BASE_IMAGE, INSTANCE_TYPES_BY_NUM_GPUS, collect_experiments
from opd.launch import gpu_overrides


def add_launch_arguments(parser):
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--profile', default='default')
    parser.add_argument('--author-name', required=True)
    parser.add_argument('--author-telegram', required=True)
    parser.add_argument('--model', type=Path, required=True, help='Checkpoint directory visible inside the job')
    parser.add_argument('--artifacts-root', type=Path, help='Shared storage outside the checkout; defaults to its sibling decomposer-artifacts')
    parser.add_argument('--lmrouter-key-file', type=Path, default=Path.home() / '.config/decomposer/lmrouter.key')
    parser.add_argument('--clearml-config', type=Path, required=True)
    parser.add_argument('--s3-config', type=Path, default=Path.home() / '.s3cfg')
    parser.add_argument('--checkpoint-s3-root', default='s3://airi-occ-artifacts/training/artifacts')
    parser.add_argument('--uv-bin', default=shutil.which('uv') or 'uv')
    parser.add_argument('--instance-type', help='Allocation instance type matching the requested GPU count')
    parser.add_argument('--num-gpus', type=int, help='Override the experiment GPU count')
    parser.add_argument('--training-gpus', type=int, help='Override the experiment training GPU count')


def resolve_paths(args, repo):
    args.artifacts_root = (args.artifacts_root or repo.parent / 'decomposer-artifacts').expanduser().resolve()
    if args.artifacts_root == repo or repo in args.artifacts_root.parents:
        raise ValueError('Artifacts must live outside the Git checkout on shared storage')
    for name in ('model', 'lmrouter_key_file', 'clearml_config', 's3_config'):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def configured_experiments(args, experiments):
    for experiment in experiments:
        changes = {key: getattr(args, key) for key in ('num_gpus', 'training_gpus')
                   if getattr(args, key) is not None}
        experiment = replace(experiment, **changes)
        gpu_overrides(list(range(experiment.num_gpus)), experiment.training_gpus)
        yield experiment


def payload(experiment, mode, staged, region, args):
    instance_type = args.instance_type or INSTANCE_TYPES_BY_NUM_GPUS.get(experiment.num_gpus)
    if not instance_type:
        raise ValueError('Pass --instance-type from mls job instance_types for this GPU count')
    gpu_overrides(list(range(experiment.num_gpus)), experiment.training_gpus)
    command = [str(staged / '.venv-opd/bin/python'), '-u', '-m',
               'opd.synth.mlspace.run_job', '--experiment', experiment.name, '--mode', mode,
               '--model', str(args.model), '--output', str(args.artifacts_root / 'sanity' / experiment.name),
               '--lmrouter-key-file', str(args.lmrouter_key_file), '--clearml-config', str(args.clearml_config),
               '--num-gpus', str(experiment.num_gpus), '--training-gpus', str(experiment.training_gpus)]
    return {'script': f'cd {shlex.quote(str(staged))} && {shlex.join(command)}',
            'job_desc': f"synth {mode} {experiment.name} #{args.author_name.lstrip('#')} @{args.author_telegram.lstrip('@')} #rnd #occ",
            'env_variables': {'WORKDIR': str(staged), 'RAY_USE_UVLOOP': '0',
                              'OPD_CHECKPOINT_S3_URI': f"{args.checkpoint_s3_root.rstrip('/')}/decomposer-{experiment.name}",
                              'OPD_S3_CONFIG': str(args.s3_config), 'UV_BIN': args.uv_bin,
                              'PYTHONPATH': f'{staged}/opd/synth/mlspace/dns:{staged}:{staged}/src:{staged}/external/verl:{staged}/external/Gym',
                              'HF_HOME': str(args.artifacts_root / 'cache/huggingface'),
                              'NO_PROXY': 'localhost,127.0.0.1,lmrouter.2a2i.org,clearml.2a2i.org',
                              'no_proxy': 'localhost,127.0.0.1,lmrouter.2a2i.org,clearml.2a2i.org'},
            'instance_type': instance_type,
            'region': region, 'type': 'binary_exp', 'shm_size_class': 'large',
            'base_image': BASE_IMAGE, 'n_workers': 1, 'processes_per_worker': 1}


def verify_runtime(staged, env_variables):
    gym = staged / 'external/Gym'
    if not (gym / 'resources_servers/workplace_assistant/utils.py').is_file():
        raise RuntimeError('Populate the pinned external/Gym submodule before submitting')
    expected = subprocess.check_output(
        ['git', '-C', str(staged), 'ls-tree', 'HEAD', 'external/Gym'], text=True).split()[2]
    actual = subprocess.check_output(['git', '-C', str(gym), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise RuntimeError(f'Gym revision mismatch: expected {expected}, got {actual}')
    subprocess.run([str(staged / '.venv-opd/bin/python'), '-c',
                    'from gyms.workplace_assistant.in_process import episode; '
                    'from opd.workplace_assistant.agent_loop import WorkplaceAgentLoop'],
                   cwd=staged, env={**os.environ, **env_variables}, check=True)


def main(mode):
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_arguments(parser)
    args = parser.parse_args()
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    resolve_paths(args, repo)
    experiments = list(configured_experiments(args, collect_experiments()))
    staged = args.artifacts_root / 'code' / commit
    if args.dry:
        for experiment in experiments:
            print(json.dumps(payload(experiment, mode, staged, 'SR008', args), indent=2))
        return
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Review and commit all changes before submitting')
    from mls.manager.job.staging import stage_repo
    from mls.manager.job.dedup import normalize_job_desc
    from mls.manager.job.utils import (get_in_progress_jobs, run_job_with_retry,
                                       training_job_api_from_profile)
    client, extra = training_job_api_from_profile(args.profile)
    active = {normalize_job_desc(j.get('job_desc', ''))
              for j in get_in_progress_jobs(client_profile=args.profile)}
    for experiment in experiments:
        output = args.artifacts_root / 'sanity' / experiment.name
        marker = output / ('baseline/manifest.json' if mode == 'baseline' else 'train/run.json')
        if marker.exists() and json.loads(marker.read_text())['status'] == 'completed':
            print(f'Skip completed: {experiment.name} {mode}')
            continue
        job = payload(experiment, mode, staged, extra['region'], args)
        descriptions = {normalize_job_desc(payload(experiment, m, staged, extra['region'],
                                                    args)['job_desc'])
                        for m in ('baseline', 'train')}
        if descriptions & active:
            print(f'Skip active: {experiment.name} {mode}')
            continue
        for directory in (output / 'baseline', output / 'train'):
            if not directory.exists():
                continue
            state = directory / ('manifest.json' if directory.name == 'baseline' else 'run.json')
            if not state.exists() or json.loads(state.read_text())['status'] != 'completed':
                raise RuntimeError(f'Preserving incomplete artifacts at {directory}; use a new experiment name')
        if repo != staged:
            stage_repo(str(repo), str(staged))
        if not (staged / '.venv-opd/bin/python').exists():
            raise RuntimeError(f'Provision the pinned runtime with bash {staged}/opd/setup.sh first')
        verify_runtime(staged, job['env_variables'])
        result = run_job_with_retry(client, job, profile=args.profile)
        output.mkdir(parents=True, exist_ok=True)
        (output / f'{mode}-submission.json').write_text(json.dumps(
            {'commit': commit, 'payload': job, 'result': result}, indent=2, default=str))
        print(json.dumps({'commit': commit, 'job_name': result.get('job_name')}))
