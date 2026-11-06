"""Exercise the real veRL adapter boundary in the OPD runtime."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip('verl')
from opd.synth import agent_loop


@pytest.mark.parametrize('message,accepted', [
    ('Decomposer requested tools instead of completing graceful shutdown', True),
    ('Invalid policy token/logprob alignment', False),
])
def test_episode_error_preserves_tokens_and_only_accepts_shutdown(tmp_path, monkeypatch, message, accepted):
    tokens = SimpleNamespace(prompt_ids=[1], response_ids=[2, 3], mask=[1, 1],
                             logprobs=[-0.2, -0.3], turns=1, extra={})
    monkeypatch.setenv('OPD_ARTIFACTS', str(tmp_path))
    monkeypatch.setattr(agent_loop, 'PolicyTokens', lambda *a, **kw: tokens)
    result = {'status': 'error', 'error_type': 'ValueError', 'error': repr(ValueError(message)),
              'cleanup_errors': [], 'score': 0.0, 'passed': False, 'parallel': False, 'copies': 0}
    monkeypatch.setattr(agent_loop, 'episode', AsyncMock(return_value=result))
    loop = object.__new__(agent_loop.SynthAgentLoop)
    loop.tokenizer = None
    loop.rollout_config = SimpleNamespace(prompt_length=32, response_length=32)
    loop.config = SimpleNamespace(synth=SimpleNamespace(worker_url='http://unused'))
    call = loop.run({}, extra_info={'task_id': 'eval-0'}, data_source='synth/validation')
    if accepted:
        output = asyncio.run(call)
        assert output.response_ids == [2, 3]
        assert output.response_mask == [1, 1]
        assert output.reward_score == 0.0
    else:
        with pytest.raises(RuntimeError, match='infrastructure failure'):
            asyncio.run(call)
    saved, = tmp_path.glob('episodes/*/trajectory.json')
    assert json.loads(saved.read_text())['result']['error'] == result['error']
