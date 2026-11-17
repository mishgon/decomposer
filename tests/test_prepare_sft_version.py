import hashlib
import json
from types import SimpleNamespace

import huggingface_hub
import pytest

from opd.wideseek import prepare_model


@pytest.mark.parametrize('version', ['1.7.0', '1.8.0'])
def test_release_version_pins_download_merge_and_provenance(tmp_path, monkeypatch, version):
    source = tmp_path / 'source'
    expected = source / version / 'merged/model.safetensors.sha256'
    expected.parent.mkdir(parents=True)
    weights = b'merged'
    expected.write_text(hashlib.sha256(weights).hexdigest())
    monkeypatch.setenv('HF_TOKEN', 'test-token')
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **kwargs:
                        SimpleNamespace(model_info=lambda repo: SimpleNamespace(sha='resolved-hf-commit')))
    downloads = []

    def download(repo, **kwargs):
        downloads.append((repo, kwargs))
        return str(source)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    monkeypatch.setattr(prepare_model.subprocess, 'check_output', lambda *a, **kw: '0')
    output = tmp_path / 'output'
    commands = []

    def merge(command, **kwargs):
        commands.append(command)
        output.mkdir()
        (output / 'model.safetensors').write_bytes(weights)

    monkeypatch.setattr(prepare_model.subprocess, 'run', merge)
    prepare_model.main(SimpleNamespace(version=version, output=output, gpu=0))
    revision = prepare_model.REVISION if version == '1.7.0' else 'resolved-hf-commit'
    assert downloads[0][1]['revision'] == revision
    assert downloads[0][1]['allow_patterns'] == ['merge.py', f'{version}/*']
    assert commands[0][-4:] == [version, str(output), '--source', str(source / version)]
    assert json.loads((output / 'opd_initialization.json').read_text()) == {
        'repo': prepare_model.REPO, 'tag': 'v' + version, 'revision': revision,
        'sha256': hashlib.sha256(weights).hexdigest()}
