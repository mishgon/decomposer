import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

spec = importlib.util.spec_from_file_location('inband_launcher',
    Path(__file__).resolve().parents[1] / 'scripts/run-inband.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_import_preserves_results_and_expands_to_four_attempts(tmp_path, monkeypatch):
    old, new = tmp_path / 'old', tmp_path / 'new'
    attempt = old / 'decomposer/task/attempt-001'
    attempt.mkdir(parents=True)
    (attempt / 'result.json').write_text('{"status":"finished"}')
    previous = {'revision': 'old', 'repetitions': 2,
                'settings': {'tasks': ['task'], 'data_sha256': 'same', 'repetitions': 2}}
    (old / 'manifest.json').write_text(json.dumps(previous))
    manifest = {'revision': 'new', 'repetitions': 4,
                'settings': {**previous['settings'], 'repetitions': 4}}
    monkeypatch.setattr(launcher, 'describe_run', AsyncMock(return_value=([], new, manifest)))
    collect = AsyncMock()
    monkeypatch.setattr(launcher, 'collect', collect)
    args = SimpleNamespace(import_run=old, output=new, n=4)
    asyncio.run(launcher.collection(args))
    assert (new / 'decomposer/task/attempt-001/result.json').read_bytes() == (attempt / 'result.json').read_bytes()
    assert json.loads((new / 'manifest.json').read_text())['imported_run']['completed_attempts'] == 1
    assert args.resume
    asyncio.run(launcher.collection(args))
    assert collect.await_count == 2
    assert launcher.describe_run.await_count == 1


def test_import_rejects_changed_task_pool(tmp_path, monkeypatch):
    old = tmp_path / 'old'
    old.mkdir()
    (old / 'manifest.json').write_text(json.dumps({'settings': {'data_sha256': 'old'}}))
    monkeypatch.setattr(launcher, 'describe_run', AsyncMock(return_value=([], tmp_path / 'new',
        {'settings': {'data_sha256': 'different'}})))
    with pytest.raises(ValueError, match='data_sha256'):
        asyncio.run(launcher.collection(SimpleNamespace(import_run=old, output=tmp_path / 'new')))
