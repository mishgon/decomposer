"""Submit committed WideSeek experiments using the installed MLSpace library."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from .experiments import (BASE_IMAGE, HOME, INSTANCE_TYPES_BY_NUM_GPUS,
                          SANITY_ROOT, STAGING_ROOT, SOURCE, collect_experiments)


def payload(experiment, mode, staged, region, author, search_url):
    command = [str(staged / '.venv-opd/bin/python'), '-u', '-m',
               'opd.wideseek.mlspace.run_job', '--experiment', experiment.name, '--mode', mode]
    return {'script': f'cd {shlex.quote(str(staged))} && {shlex.join(command)}',
            'job_desc': f'wideseek {mode} {experiment.name} #{author} @js0n_statham #rnd #occ',
            'env_variables': {'HOME': str(HOME), 'WORKDIR': str(staged),
                              'RAY_USE_UVLOOP': '0',
                              'WS_SEARCH_URL': search_url, 'WS_TASK_SOURCE': str(SOURCE),
                              'OPD_CHECKPOINT_S3_URI': f's3://airi-occ-artifacts/training/artifacts/matrosov-{experiment.name}',
                              'OPD_S3_CONFIG': str(HOME / '.s3cfg'),
                              'UV_BIN': '/home/jovyan/.local/bin/uv',
                              'PYTHONPATH': f'{staged}/opd/synth/mlspace/dns:{staged}:{staged}/src:{staged}/external/verl:{staged}/external/Gym',
                              'HF_HOME': '/home/jovyan/.cache/huggingface',
                              'NO_PROXY': 'localhost,127.0.0.1,lmrouter.2a2i.org,clearml.2a2i.org',
                              'no_proxy': 'localhost,127.0.0.1,lmrouter.2a2i.org,clearml.2a2i.org'},
            'instance_type': INSTANCE_TYPES_BY_NUM_GPUS[experiment.num_gpus],
            'region': region, 'type': 'binary_exp', 'shm_size_class': 'large',
            'base_image': BASE_IMAGE, 'n_workers': 1, 'processes_per_worker': 1}


def verify_runtime(staged, env_variables):
    if not SOURCE.is_file():
        raise RuntimeError(f'Prepare pinned WideSeek tasks first: {SOURCE}')
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
                    'from opd.wideseek.agent_loop import WideSeekAgentLoop; '
                    'from opd.workplace_assistant.agent_loop import WorkplaceAgentLoop'],
                   cwd=staged, env={**os.environ, **env_variables}, check=True)


def main(mode):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--search-url', required=True, help='Wiki-2018 retrieval URL reachable inside the job')
    parser.add_argument('--profile', default='default')
    parser.add_argument('--author-name', default='matrosov')
    args = parser.parse_args()
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    staged = STAGING_ROOT / commit
    if args.dry:
        for experiment in collect_experiments():
            print(json.dumps(payload(experiment, mode, staged, 'SR008', args.author_name, args.search_url), indent=2))
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
    for experiment in collect_experiments():
        output = SANITY_ROOT / experiment.name
        marker = output / ('baseline/manifest.json' if mode == 'baseline' else 'train/run.json')
        if marker.exists() and json.loads(marker.read_text())['status'] == 'completed':
            print(f'Skip completed: {experiment.name} {mode}')
            continue
        job = payload(experiment, mode, staged, extra['region'], args.author_name, args.search_url)
        descriptions = {normalize_job_desc(payload(experiment, m, staged, extra['region'],
                                                    args.author_name, args.search_url)['job_desc'])
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
