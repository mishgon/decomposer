import hashlib
import json
from types import SimpleNamespace

import pytest

from opd.wideseek.local import check_inputs


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setenv('RAY_TMPDIR', '/mnt/share14T-2/matrosov/ray')
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    source = tmp_path / 'tasks.jsonl'
    source.write_text('task data')
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    (baseline / 'manifest.json').write_text(json.dumps({
        'student_model': str(model.resolve()),
        'status': 'completed', 'model': 'vllm/qwen_3_5_4b_non_thinking',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}))
    return SimpleNamespace(model=model, source=source, baseline=baseline)


def test_rejects_actual_failed_ray_path_before_launch(inputs, monkeypatch):
    monkeypatch.setenv('RAY_TMPDIR', '/mnt/share14T-2/matrosov/wideseek/tmp/ray')
    with pytest.raises(ValueError, match='RAY_TMPDIR is too long'):
        check_inputs(inputs)


def test_accepts_short_path_and_matching_completed_baseline(inputs):
    check_inputs(inputs)


def test_rejects_baseline_source_mismatch(inputs):
    inputs.source.write_text('changed tasks')
    with pytest.raises(ValueError, match='source differ'):
        check_inputs(inputs)


def test_rejects_unfinished_baseline(inputs):
    p = inputs.baseline / 'manifest.json'
    data = json.loads(p.read_text())
    data['status'] = 'interrupted'
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='completed coordinator'):
        check_inputs(inputs)


def test_rejects_baseline_from_different_starting_checkpoint(inputs):
    p = inputs.baseline / 'manifest.json'
    data = json.loads(p.read_text())
    data['student_model'] = '/old/stock-model'
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='selected starting checkpoint'):
        check_inputs(inputs)
