from collections import Counter
from contextlib import contextmanager
import asyncio
import json
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pandas as pd
import pytest

from gyms.wideseek.prepare import normalize
from opd.mix import smoke, smoke_data
from opd.wideseek.prepare import check


@pytest.fixture
def source(tmp_path):
    tasks = [normalize({'question': f'{split} question {i}', 'answer': 'PRIVATE REFERENCE'}, split, i)
             for split in smoke_data.SOURCES for i in range(5)]
    path = tmp_path / 'tasks.jsonl'
    path.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    return path


def test_smoke_has_four_tasks_and_exactly_two_episodes_each(source, tmp_path):
    output = tmp_path / 'data'
    manifest = smoke_data.prepare(source, output)
    assert check(output) == manifest
    assert manifest['training_sources'] == dict.fromkeys(smoke_data.SOURCES, 1)
    rows = pd.read_parquet(output / 'train.parquet').to_dict('records')
    assert Counter(r['extra_info']['task_id'] for r in rows) == dict.fromkeys(manifest['task_ids'], 2)
    assert Counter(r['data_source'] for r in rows) == {'deepdive/train': 4, 'wideseek/train': 4}
    assert all(r['agent_name'] == 'wideseek_decomposer' for r in rows)
    assert all(r['reward_model']['ground_truth'] == '' for r in rows)
    assert all('PRIVATE REFERENCE' not in str(r['prompt']) for r in rows)
    second = smoke_data.prepare(source, tmp_path / 'second')
    assert manifest['task_ids'] == second['task_ids']
    assert manifest['heldout_tasks'] == manifest['workplace_validation_tasks'] == 0


def test_smoke_excludes_frozen_holdouts_and_duplicate_questions(source, tmp_path, monkeypatch):
    tasks = [json.loads(line) for line in source.read_text().splitlines()]
    panel = tmp_path / 'panel.json'
    panel.write_text(json.dumps({'heldout': tasks[:4]}))
    monkeypatch.setattr(smoke_data, 'DEPTH_VALIDATION_PANEL', panel)
    # The only remaining QA-RL question also appears in QA-SFT; choose a different QA-SFT task.
    tasks[5]['question_sha256'] = tasks[4]['question_sha256']
    source.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    output = tmp_path / 'data'
    smoke_data.prepare(source, output)
    selected = [json.loads(line) for line in (output / 'tasks.jsonl').read_text().splitlines()]
    assert selected[0]['task_id'] == tasks[4]['task_id']
    assert len({t['question_sha256'] for t in selected}) == 4


def test_missing_split_fails_before_writing(source, tmp_path):
    source.write_text('\n'.join(line for line in source.read_text().splitlines()
                                if json.loads(line)['source'] != 'width'))
    with pytest.raises(ValueError, match='eligible task from width'):
        smoke_data.prepare(source, tmp_path / 'data')
    assert not (tmp_path / 'data').exists()


def test_smoke_uses_production_trainer_without_a_fake_baseline(tmp_path, monkeypatch):
    args = smoke.create_parser(smoke=True).parse_args([
        '--source', str(tmp_path / 'tasks'), '--output', str(tmp_path / 'run'),
        '--model', '/models/sft-v1.7.0', '--gpus', '4', '5'])
    launch = AsyncMock()
    monkeypatch.setattr(smoke, 'train', launch)
    monkeypatch.setattr(smoke, 'check_inputs', lambda _: None)
    monkeypatch.setattr(smoke.subprocess, 'run', lambda *a, **kw: None)

    @contextmanager
    def reserve(*_):
        yield

    monkeypatch.setattr(smoke, 'reserve_gpus', reserve)
    asyncio.run(smoke.main(args))
    launch.assert_awaited_once_with(args, smoke.OVERRIDES, prepare_data=smoke_data.prepare, smoke=True)
    assert args.baseline is None
    assert 'rollout.total_rollout_steps=8' in smoke.OVERRIDES
    assert not any(o.startswith(('actor_rollout_ref.', 'distillation.')) for o in smoke.OVERRIDES)


def test_full_run_still_requires_a_baseline():
    with pytest.raises(SystemExit):
        smoke.create_parser().parse_args(['--source', '/tasks', '--output', '/run', '--gpus', '4', '5'])


@pytest.mark.parametrize('hits', [[], [{'title': 'Paris'}]])
def test_preflight_probes_the_researchers_actual_retrieval_api(tmp_path, monkeypatch, hits):
    from opd.wideseek import preflight
    from transformers import AutoTokenizer

    monkeypatch.setenv('AGENTIC_RAG_API_KEY', 'test-key')
    monkeypatch.setenv('AGENTIC_RAG_URL', 'https://retrieval.invalid/rag')
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'results': hits})
    http = AsyncMock()
    http.__aenter__.return_value = http
    http.post.return_value = response
    http.get.return_value = response
    monkeypatch.setattr(preflight.httpx, 'AsyncClient', lambda **kw: http)
    worker = SimpleNamespace(assistants=SimpleNamespace(get_graph=AsyncMock()))
    monkeypatch.setattr(preflight, 'get_client', lambda **kw: worker)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda _: SimpleNamespace(
        apply_chat_template=lambda *a, **kw: [1, 2]))
    monkeypatch.setattr(preflight, 'score', AsyncMock(return_value=[0, -.5]))
    monkeypatch.setattr(preflight, 'model', lambda _: SimpleNamespace(
        ainvoke=AsyncMock(return_value=SimpleNamespace(content='OK'))))
    monkeypatch.setattr(preflight, 'close_model', AsyncMock())
    call = preflight.check('/model', tmp_path, worker_url='http://worker')
    if hits:
        result = asyncio.run(call)
        assert result['retrieval'] == 'ready'
    else:
        with pytest.raises(ValueError, match='no passages'):
            asyncio.run(call)
        preflight.score.assert_not_awaited()
    http.post.assert_awaited_once_with('https://retrieval.invalid/rag/search',
        headers={'Authorization': 'Bearer test-key'},
        json={'query': 'Paris', 'limit': 1, 'source': preflight.agentic_rag.SOURCE})
