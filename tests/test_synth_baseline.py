import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from evals.synth import run


@pytest.mark.parametrize('error_type,cleanup,completed', [
    (None, [], True),
    ('GraphRecursionError', [], True),
    ('APIConnectionError', [], False),
    (None, ['worker still running'], False),
])
def test_baseline_preserves_task_failures_but_rejects_infrastructure(tmp_path, monkeypatch, error_type, cleanup, completed):
    @asynccontextmanager
    async def server(*args, **kwargs):
        yield 'http://unused'

    policy = SimpleNamespace(http_client=Mock(), http_async_client=SimpleNamespace(aclose=AsyncMock()))
    monkeypatch.setattr(run, 'agent_server', server)
    monkeypatch.setattr(run, 'create_model', lambda _: policy)
    monkeypatch.setattr(run, 'episode', AsyncMock(return_value={
        'task_id': 'eval-0', 'status': 'error' if error_type or cleanup else 'finished',
        'error_type': error_type, 'cleanup_errors': cleanup,
        'score': 0.25, 'passed': False, 'parallel': False, 'copies': 1,
    }))
    args = SimpleNamespace(tasks=1, repetitions=1, concurrency=1, output=tmp_path / 'baseline',
                           model='vllm/qwen_3_5_4b_non_thinking', split='eval',
                           student_model=tmp_path / 'model', worker_python=None)
    if completed:
        asyncio.run(run.main(args))
        assert json.loads((args.output / 'summary.json').read_text())['mean_score'] == 0.25
    else:
        with pytest.raises(ExceptionGroup, match='TaskGroup'):
            asyncio.run(run.main(args))
        assert not (args.output / 'summary.json').exists()
    manifest = json.loads((args.output / 'manifest.json').read_text())
    assert manifest['status'] == ('completed' if completed else 'interrupted')
    assert manifest['student_model'] == str(args.student_model.resolve())
    assert len(manifest['episodes']) == 1
    policy.http_client.close.assert_called_once()
    policy.http_async_client.aclose.assert_awaited_once()


@pytest.mark.parametrize('baseline_model', [None, '/some/other/checkpoint'])
def test_opd_rejects_baseline_without_matching_checkpoint(tmp_path, baseline_model):
    from opd.synth import run as training
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    (baseline / 'manifest.json').write_text(json.dumps({
        'status': 'completed', 'model': 'vllm/qwen_3_5_4b_non_thinking',
        'student_model': baseline_model,
    }))
    args = SimpleNamespace(baseline=baseline, model=tmp_path / 'model', gpus=[0, 1])
    with pytest.raises(ValueError, match='same student checkpoint path'):
        asyncio.run(training.main(args, []))
