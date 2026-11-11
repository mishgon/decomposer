import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from opd.workplace_assistant import preflight


@pytest.mark.parametrize('error', [None, TimeoutError('timeout'), RuntimeError('model missing')])
def test_configured_worker_is_checked_and_closed(monkeypatch, error):
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


def test_synth_preflight_failure_happens_before_creating_training_output(tmp_path, monkeypatch):
    import json
    from opd.synth import run
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
