import json

import pytest

from opd.wideseek.fast_runtime import DIRECTORIES, environment, restore


def test_restored_overlay_is_verified_and_does_not_mutate_environment(tmp_path, monkeypatch):
    source = tmp_path / 'backup'
    for name in DIRECTORIES:
        (source / name).mkdir(parents=True)
        (source / name / 'kernel.py').write_text('original')
    output = tmp_path / 'bundle'
    restore(source, output)
    monkeypatch.setenv('PYTHONPATH', '/original')
    env = environment(output)
    assert env['PYTHONPATH'].endswith(':/original')
    assert env['FLA_TILELANG'] == '0'
    import os
    assert os.environ['PYTHONPATH'] == '/original'
    (output / DIRECTORIES[0] / 'kernel.py').write_text('changed')
    with pytest.raises(ValueError, match='contents changed'):
        environment(output)


def test_rejects_wrong_runtime_profile(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'versions': {}, 'python': '3.12'}))
    with pytest.raises(ValueError, match='profile'):
        environment(tmp_path)
