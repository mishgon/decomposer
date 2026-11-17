import hashlib
import json
import sys

import pytest

from evals.wideseek.occtopus import seen


def test_pool_is_frozen_without_importing_previous_attempts(tmp_path):
    source = tmp_path / 'pool.jsonl'
    source.write_text('{"task_id":"depth-1"}\n{"task_id":"width-2"}\n')
    recipe = {'pool_tasks': 2, 'pool_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    root = tmp_path / 'new'
    root.mkdir()
    assert seen.prepare_pool(source, root, recipe).read_bytes() == source.read_bytes()
    source.write_text(source.read_text().replace('depth-1', 'depth-2'))
    with pytest.raises(ValueError, match='frozen recipe'):
        seen.prepare_pool(source, root, recipe)
    assert not (root / 'eval').exists()


def test_pool_rejects_duplicate_tasks(tmp_path):
    source = tmp_path / 'pool.jsonl'
    source.write_text('{"task_id":"depth-1"}\n' * 2)
    with pytest.raises(ValueError, match='duplicate IDs'):
        seen.prepare_pool(source, tmp_path,
                          {'pool_tasks': 2, 'pool_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})


@pytest.mark.parametrize('version', ['1.7.0', '1.8.0'])
def test_launch_uses_requested_checkpoint_and_same_fifteen_minute_recipe(tmp_path, monkeypatch, version):
    root = tmp_path / ('run-' + version)
    model = tmp_path / 'models' / ('decomposer-manager-sft-v' + version)
    model.mkdir(parents=True)
    weights = b'verified merged weights'
    (model / 'model.safetensors').write_bytes(weights)
    (model / 'opd_initialization.json').write_text(json.dumps(
        {'tag': 'v' + version, 'sha256': hashlib.sha256(weights).hexdigest()}))
    monkeypatch.setattr(seen, 'prepare_pool', lambda *args: root / 'tasks.jsonl')
    command = []
    monkeypatch.setattr(seen.os, 'execv', lambda _, argv: command.extend(argv))
    monkeypatch.setattr(sys, 'argv', ['seen', '--root', str(root), '--models', str(tmp_path / 'models'),
        '--data', '/frozen/pool.jsonl', '--inference-python', '/inference/python', '--sft-version', version])
    seen.main()
    for option, value in [('--model', str(model)), ('--episode-timeout', '900'),
                          ('--concurrency', '256'), ('-n', '4'), ('--limit', '4783'),
                          ('--max-pass-rate', '0.5'), ('--worker-port', '0')]:
        assert command[command.index(option) + 1] == value
    assert '--import-run' not in command
    assert '--require-success' not in command
    assert json.loads((root / 'recipe.json').read_text())['sft_version'] == version
    monkeypatch.setattr(sys, 'argv', sys.argv[:-1] + ['1.9.0'])
    with pytest.raises(ValueError, match='Cannot change checkpoint'):
        seen.main()
