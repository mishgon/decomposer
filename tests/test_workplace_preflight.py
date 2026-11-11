import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from opd.workplace_assistant import preflight


@pytest.mark.parametrize('error', [None, TimeoutError('timeout'), RuntimeError('model missing')])
def test_calibrated_worker_is_checked_and_closed(monkeypatch, error):
    worker = SimpleNamespace(ainvoke=AsyncMock(side_effect=error),
                             http_client=Mock(), http_async_client=SimpleNamespace(aclose=AsyncMock()))
    factory = Mock(return_value=worker)
    monkeypatch.setattr(preflight, 'create_model', factory)
    if error:
        with pytest.raises(RuntimeError, match='Workplace validation worker preflight failed'):
            asyncio.run(preflight.check())
    else:
        asyncio.run(preflight.check())
    factory.assert_called_once_with(preflight.WORKER_MODEL)
    worker.ainvoke.assert_awaited_once_with('Reply OK.', max_tokens=16)
    worker.http_client.close.assert_called_once()
    worker.http_async_client.aclose.assert_awaited_once()


@pytest.mark.parametrize('gym', ['synth', 'wideseek'])
def test_preflight_failure_happens_before_creating_training_output(tmp_path, monkeypatch, gym):
    import importlib
    import json
    run = importlib.import_module(f'opd.{gym}.run')
    monkeypatch.delenv('OPD_FAST_RUNTIME', raising=False)
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    (baseline / 'manifest.json').write_text(json.dumps({
        'status': 'completed', 'model': 'vllm/qwen_3_5_4b_non_thinking',
        'student_model': str(tmp_path / 'model')}))
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    output = tmp_path / 'train'
    args = SimpleNamespace(baseline=baseline, model=model, output=output,
                           gpus=[0, 1], training_gpus=1, resume_from=None, workplace_validation=True)
    monkeypatch.setattr(preflight, 'check', AsyncMock(side_effect=RuntimeError('worker unavailable')))
    server = Mock()
    monkeypatch.setattr(run, 'agent_server', server)
    with pytest.raises(RuntimeError, match='worker unavailable'):
        asyncio.run(run.main(args, []))
    server.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize('training_gpus', [1, 2])
@pytest.mark.parametrize('gpus', [[4, 5, 6], [4, 5, 6, 7]])
def test_wideseek_gpu_split_is_explicit(tmp_path, monkeypatch, training_gpus, gpus):
    import json
    from opd.wideseek import run
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (baseline / 'manifest.json').write_text(json.dumps({'status': 'completed',
        'model': 'vllm/qwen_3_5_4b_non_thinking', 'student_model': str(model)}))
    args = SimpleNamespace(baseline=baseline, model=model, output=tmp_path/'output',
        gpus=gpus, training_gpus=training_gpus, resume_from=None, source=tmp_path/'source')
    monkeypatch.delenv('OPD_FAST_RUNTIME', raising=False)
    monkeypatch.setattr(preflight, 'check', AsyncMock())
    monkeypatch.setattr(run.subprocess, 'run', Mock())
    monkeypatch.setattr(run, 'prepare', Mock(return_value={'source_sha256': None}))
    import sys
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(side_effect=RuntimeError('stop before GPU startup')))))
    with pytest.raises(RuntimeError, match='stop before GPU startup'):
        asyncio.run(run.main(args, []))
    saved = json.loads((args.output/'run.json').read_text())
    assert f'trainer.n_gpus_per_node={training_gpus}' in saved['overrides']
    assert f'rollout.n_gpus_per_node={len(gpus)-training_gpus}' in saved['overrides']
    # Check invalid rank counts before any service request.
    args.training_gpus = len(gpus)
    preflight.check.reset_mock()
    with pytest.raises(ValueError, match='at least one'):
        asyncio.run(run.main(args, []))
    preflight.check.assert_not_awaited()
