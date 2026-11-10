import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from opd.wideseek.mlspace import jobs, run_job
from opd.wideseek.mlspace.experiments import collect_experiments


@pytest.mark.parametrize('mode', ['baseline', 'train'])
def test_dry_run_forwards_explicit_settings(mode, monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['launcher', '--dry', '--author-name', 'researcher',
        '--author-telegram', '@researcher', '--model', '/shared/model',
        '--clearml-config', '/shared/config', '--source', '/shared/tasks.jsonl',
        '--search-url', 'http://retrieval:18080', '--num-gpus', '4', '--training-gpus', '2',
        '--instance-type', 'allocation-four-gpu'])
    monkeypatch.setattr(jobs.subprocess, 'check_output', Mock(side_effect=['/shared/repo\n', 'abc\n']))
    jobs.main(mode)
    payload = json.loads(capsys.readouterr().out)
    assert payload['job_desc'].startswith(f'wideseek {mode} ')
    assert '#researcher @researcher' in payload['job_desc']
    assert 'opd.wideseek.mlspace.run_job' in payload['script']
    assert '--num-gpus 4 --training-gpus 2 --source /shared/tasks.jsonl' in payload['script']
    assert payload['env_variables']['WS_TASK_SOURCE'] == '/shared/tasks.jsonl'
    assert payload['env_variables']['WS_SEARCH_URL'] == 'http://retrieval:18080'
    assert 'HOME' not in payload['env_variables']


@pytest.mark.parametrize('count,trainers', [(1, 1), (3, 2), (4, 2), (8, 4)])
def test_runtime_forwards_layout(count, trainers, tmp_path, monkeypatch):
    experiment, = collect_experiments()
    output = tmp_path / 'output'
    (output / 'baseline').mkdir(parents=True)
    (output / 'baseline/manifest.json').write_text('{"status":"completed"}')
    (tmp_path / 'config.json').touch()
    monkeypatch.setattr('sys.argv', ['run_job', '--experiment', experiment.name, '--mode', 'train',
        '--output', str(output), '--model', str(tmp_path), '--lmrouter-key-file', '/shared/key',
        '--clearml-config', '/shared/config', '--source', '/shared/tasks.jsonl',
        '--num-gpus', str(count), '--training-gpus', str(trainers)])
    monkeypatch.setattr(run_job, 'configure_network', Mock())
    monkeypatch.setattr(run_job, 'preflight', Mock())
    monkeypatch.setattr(run_job.os, 'chdir', Mock())
    monkeypatch.setattr('opd.checkpoint_offload.check_access', Mock())
    monkeypatch.setenv('OPD_CHECKPOINT_S3_URI', 's3://test/run')
    monkeypatch.setenv('WS_SEARCH_URL', 'http://retrieval:18080')
    monkeypatch.setenv('PATH', run_job.os.environ['PATH'])
    client = MagicMock()
    client.return_value.__enter__.return_value.post.return_value.json.return_value = {'result': [['passage']]}
    monkeypatch.setitem(__import__('sys').modules, 'httpx', SimpleNamespace(Client=client))
    command = Mock()
    monkeypatch.setattr(run_job.subprocess, 'run', command)
    run_job.main()
    train = command.call_args.args[0]
    assert train[train.index('--gpus') + 1:train.index('--training-gpus')] == list(map(str, range(count)))
    assert train[train.index('--training-gpus') + 1] == str(trainers)
    assert train[train.index('--source') + 1] == '/shared/tasks.jsonl'
