"""Exercise both validation adapters with real veRL output schemas, without a GPU."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip('verl')
from opd.decomposer_demo import agent_loop as demo
from opd.workplace_assistant import agent_loop as workplace


@pytest.mark.parametrize('adapter', ['demo', 'workplace'])
@pytest.mark.parametrize('error_type,error,worker_errors,accepted', [
    ('GraphRecursionError', 'GraphRecursionError: exhausted', [], True),
    ('ValueError', 'ValueError: Decomposer requested tools instead of completing graceful shutdown', [], True),
    ('WorkerError', 'WorkerError: exhausted', ['GraphRecursionError: exhausted'], True),
    ('WorkerError', 'WorkerError: offline', ['APIConnectionError: offline'], False),
    ('TimeoutError', 'TimeoutError: deadline', ['APIConnectionError: offline'], False),
    ('ValueError', 'ValueError: Invalid policy token/logprob alignment', [], False),
])
def test_expected_failures_finish_and_infrastructure_errors_surface(
        tmp_path, monkeypatch, adapter, error_type, error, worker_errors, accepted):
    module, cls = (demo, demo.DemoAgentLoop) if adapter == 'demo' else (workplace, workplace.WorkplaceAgentLoop)
    monkeypatch.setenv('OPD_ARTIFACTS', str(tmp_path))
    monkeypatch.setenv('OPD_ROOT', str(tmp_path))
    panel = tmp_path / 'gyms/workplace_assistant/validation20.json'
    panel.parent.mkdir(parents=True)
    panel.write_text(json.dumps([{'id': 1}]))
    worker = SimpleNamespace(http_client=Mock(), http_async_client=SimpleNamespace(aclose=AsyncMock()))
    if adapter == 'workplace':
        monkeypatch.setattr(module, 'create_model', lambda _: worker)
    tokens = SimpleNamespace(prompt_ids=[1], response_ids=[2, 3], mask=[1, 1],
                             logprobs=[-.2, -.3], turns=1, extra={})
    monkeypatch.setattr(module, 'PolicyTokens', lambda *a, **kw: tokens)
    result = dict(status='error', error_type=error_type, error=error,
                  worker_errors=worker_errors, score=1., passed=False)
    monkeypatch.setattr(module, 'episode', AsyncMock(return_value=result))
    loop = object.__new__(cls)
    loop.tokenizer = None
    loop.rollout_config = SimpleNamespace(prompt_length=32, response_length=32)
    source = 'decomposer-demo/ru' if adapter == 'demo' else 'workplace-assistant/validation20'
    call = loop.run({}, data_source=source, extra_info={'task_id': '1', 'language': 'ru'})
    if accepted:
        output = asyncio.run(call)
        assert output.reward_score == 0
        assert output.response_ids == [2, 3]
        assert output.response_mask == [1, 1]
    else:
        with pytest.raises(RuntimeError, match='validation'):
            asyncio.run(call)
    path, = tmp_path.glob('*-episodes/*/result.json')
    assert json.loads(path.read_text())['error'] == error
