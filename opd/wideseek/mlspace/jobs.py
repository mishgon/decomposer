"""Submit committed WideSeek experiments using the installed MLSpace library."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from .experiments import collect_experiments
from opd.synth.mlspace.jobs import (add_launch_arguments, configured_experiments,
                                    resolve_paths, payload as synth_payload)


def payload(experiment, mode, staged, region, args):
    job = synth_payload(experiment, mode, staged, region, args, gym='wideseek',
                        extra_args=['--source', str(args.source)])
    job['env_variables'].update(WS_SEARCH_URL=args.search_url, WS_TASK_SOURCE=str(args.source))
    return job


def verify_runtime(staged, env_variables):
    source = Path(env_variables['WS_TASK_SOURCE'])
    if not source.is_file():
        raise RuntimeError(f'Prepare pinned WideSeek tasks first: {source}')
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
    add_launch_arguments(parser)
    parser.add_argument('--search-url', required=True, help='Wiki-2018 retrieval URL reachable inside the job')
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    resolve_paths(args, repo)
    args.source = args.source.expanduser().resolve()
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
