import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import pytest
from unittest.mock import Mock

from opd.synth.mlspace import jobs, run_job
from opd.synth.mlspace.experiments import collect_experiments


def test_runtime_rejects_empty_gym_before_importing(tmp_path, monkeypatch):
    (tmp_path / 'external/Gym').mkdir(parents=True)
    command = Mock()
    monkeypatch.setattr(jobs.subprocess, 'run', command)
    with pytest.raises(RuntimeError, match='Populate'):
        jobs.verify_runtime(tmp_path, {})
    command.assert_not_called()


def test_runtime_checks_revision_and_imports_with_job_environment(tmp_path, monkeypatch):
    utils = tmp_path / 'external/Gym/resources_servers/workplace_assistant/utils.py'
    utils.parent.mkdir(parents=True)
    utils.touch()
    monkeypatch.setattr(jobs.subprocess, 'check_output', Mock(side_effect=[
        '160000 commit pinned\texternal/Gym\n', 'wrong\n',
        '160000 commit pinned\texternal/Gym\n', 'pinned\n']))
    command = Mock()
    monkeypatch.setattr(jobs.subprocess, 'run', command)
    with pytest.raises(RuntimeError, match='revision mismatch'):
        jobs.verify_runtime(tmp_path, {})
    command.assert_not_called()
    jobs.verify_runtime(tmp_path, {'PYTHONPATH': 'job-path'})
    assert command.call_args.kwargs['env']['PYTHONPATH'] == 'job-path'
    assert 'WorkplaceAgentLoop' in command.call_args.args[0][-1]


def test_dry_run_never_submits_or_reads_credentials(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['launcher', '--dry', '--author-name', 'researcher', '--author-telegram', 'researcher',
        '--model', '/shared/model', '--clearml-config', '/shared/clearml.conf'])
    monkeypatch.setattr(jobs.subprocess, 'check_output', Mock(side_effect=['/repo\n', 'abc123\n']))
    jobs.main('train')
    payload = json.loads(capsys.readouterr().out)
    assert payload['n_workers'] == payload['processes_per_worker'] == 1
    assert '#researcher @researcher' in payload['job_desc']
    assert payload['env_variables']['RAY_USE_UVLOOP'] == '0'
    assert payload['env_variables']['OPD_CHECKPOINT_S3_URI'].startswith('s3://airi-occ-artifacts/training/artifacts/decomposer-')
    assert '.2gpu.' in payload['instance_type']
    assert '/code/abc123' in payload['script']
    assert not any('KEY' in k or 'SECRET' in k for k in payload['env_variables'])


def test_network_configuration_removes_proxies(monkeypatch):
    monkeypatch.setattr(run_job.socket, 'getaddrinfo', Mock(return_value=[(2, 1, 6, '', ('10.0.0.1', 443))]))
    command = Mock()
    monkeypatch.setattr(run_job.subprocess, 'run', command)
    monkeypatch.setenv('https_proxy', 'https://unused.invalid')
    monkeypatch.setenv('ALL_PROXY', 'socks5://unused.invalid')
    run_job.configure_network()
    command.assert_not_called()
    assert 'https_proxy' not in run_job.os.environ
    assert 'ALL_PROXY' not in run_job.os.environ
    assert 'lmrouter.2a2i.org' in run_job.os.environ['NO_PROXY']


def test_dns_patch_inherited_by_child_python():
    dns = Path(run_job.__file__).parent / 'dns'
    env = {**os.environ, 'PYTHONPATH': str(dns)}
    script = """
import socket, subprocess, sys
assert socket.getaddrinfo('clearml.2a2i.org', 443)[0][4][0] == '176.108.242.226'
assert socket.getaddrinfo(b'lmrouter.2a2i.org', 443)[0][4][0] == '176.108.242.226'
assert socket.getaddrinfo('127.0.0.1', 443)[0][4][0] == '127.0.0.1'
subprocess.run([sys.executable, '-c',
    "import socket; assert socket.getaddrinfo('lmrouter.2a2i.org', 443)[0][4][0] == '176.108.242.226'"], check=True)
"""
    subprocess.run([sys.executable, '-c', script], env=env, check=True)


def test_baseline_job_uses_same_experiment():
    experiment, = collect_experiments()
    parser = argparse.ArgumentParser()
    jobs.add_launch_arguments(parser)
    args = parser.parse_args(['--author-name', 'researcher', '--author-telegram', 'researcher',
        '--model', '/shared/model', '--clearml-config', '/shared/clearml.conf'])
    jobs.resolve_paths(args, Path('/shared/repo'))
    payload = jobs.payload(experiment, 'baseline', Path('/staged'), 'region', args)
    assert '--mode baseline' in payload['script']
    assert experiment.rollout_steps == 512


@pytest.mark.parametrize('count,trainers', [(1, 1), (2, 1), (4, 2), (8, 4)])
def test_job_forwards_gpu_layout(tmp_path, monkeypatch, count, trainers):
    experiment, = collect_experiments()
    assert experiment.num_gpus == 2
    output = tmp_path / experiment.name
    (output / 'baseline').mkdir(parents=True)
    (output / 'baseline/manifest.json').write_text('{"status":"completed"}')
    (tmp_path / 'config.json').touch()
    monkeypatch.setattr(run_job, 'configure_network', Mock())
    monkeypatch.setattr(run_job, 'preflight', Mock())
    monkeypatch.setattr('opd.checkpoint_offload.check_access', Mock())
    monkeypatch.setenv('OPD_CHECKPOINT_S3_URI', 's3://test/run')
    monkeypatch.setattr(run_job.os, 'chdir', Mock())
    monkeypatch.setenv('PATH', os.environ['PATH'])
    monkeypatch.setattr('sys.argv', ['run_job', '--experiment', experiment.name, '--mode', 'train',
        '--output', str(output), '--model', str(tmp_path), '--lmrouter-key-file', '/shared/key',
        '--clearml-config', '/shared/clearml.conf', '--num-gpus', str(count), '--training-gpus', str(trainers)])
    command = Mock()
    monkeypatch.setattr(run_job.subprocess, 'run', command)
    run_job.main()
    run_job.preflight.assert_called_once_with(output, tmp_path, Path('/shared/key'),
                                               Path('/shared/clearml.conf'), workplace_validation=True)
    train = command.call_args.args[0]
    assert train[train.index('--gpus') + 1:train.index('--gpus') + 1 + count] == list(map(str, range(count)))
    assert train[train.index('--training-gpus') + 1] == str(trainers)
    assert '--gpu' not in train
