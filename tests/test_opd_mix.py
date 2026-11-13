import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gyms.wideseek.prepare import normalize
from opd.mix import prepare as mix
from opd.wideseek import prepare as wide


@pytest.fixture
def depth_source(tmp_path, monkeypatch):
    from opd.workplace_assistant import prepare as workplace

    tasks = [normalize({'question': f'Depth {i}', 'answer': 'Reference'}, 'depth', i)
             for i in range(40)]
    source = tmp_path / 'depth.jsonl'
    source.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    panel = tmp_path / 'panel.json'
    panel.write_text(json.dumps({name: [{k: tasks[i][k] for k in ('task_id', 'question_sha256')}
                                      for i in ids]
                                for name, ids in [('train_probe', range(4)), ('heldout', range(4, 12))]}))
    monkeypatch.setattr(wide, 'DEPTH_VALIDATION_PANEL', panel)
    ids = json.loads((Path(__file__).resolve().parents[1] /
                      'gyms/workplace_assistant/validation20.json').read_text())
    monkeypatch.setattr(workplace, 'validation_rows', lambda: [
        {'data_source': 'workplace-assistant/validation20', 'agent_name': 'workplace_decomposer',
         'prompt': [{'role': 'user', 'content': f'Task {i}'}],
         'reward_model': {'style': 'rule', 'ground_truth': str(i)},
         'extra_info': {'task_id': str(i)}} for i in ids])
    return source


def test_depth_control_preserves_data_and_validation(depth_source, tmp_path):
    original = wide.prepare(depth_source, tmp_path / 'wide')
    prepared = mix.prepare(depth_source, tmp_path / 'mix')
    assert wide.check(tmp_path / 'mix') == prepared
    assert prepared.pop('training_mixture') == [
        {'gym': 'wideseek', 'split': 'depth', 'weight': 1.0}]
    prepared['files'].pop('mixture.json')
    assert prepared == original
    for name in original['files']:
        assert (tmp_path / 'wide' / name).read_bytes() == (tmp_path / 'mix' / name).read_bytes()
    (tmp_path / 'mix/mixture.json').write_text('{}')
    with pytest.raises(ValueError, match='Prepared data changed: mixture.json'):
        wide.check(tmp_path / 'mix')


def test_rejects_other_splits_before_creating_data(tmp_path):
    source = tmp_path / 'width.jsonl'
    source.write_text(json.dumps(normalize({'question': 'Question', 'answer': 'Answer'}, 'width', 0)))
    with pytest.raises(ValueError, match='depth tasks only'):
        mix.prepare(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_rejects_unsupported_mixture(depth_source, tmp_path, monkeypatch):
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps({'training_sources': [
        {'gym': 'wideseek', 'split': 'depth', 'weight': 0.5},
        {'gym': 'deepdive', 'split': 'qa_rl', 'weight': 0.5}]}))
    monkeypatch.setattr(mix, 'RECIPE', recipe)
    with pytest.raises(ValueError, match='Only the 100% WideSeek-depth mixture'):
        mix.prepare(depth_source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_run_preserves_arguments_and_uses_mixture_preparer(monkeypatch):
    from opd.mix import run

    parser = run.create_parser()
    args, overrides = parser.parse_known_args([
        '--source', '/data/depth.jsonl', '--baseline', '/run/baseline',
        '--model', '/models/sft', '--output', '/run/train', '--gpus', '4', '5', '6',
        '--training-gpus', '2', '--resume-from', '/previous/global_step_31',
        'trainer.experiment_name=mix-test'])
    launch = AsyncMock()
    monkeypatch.setattr(run, 'run', launch)
    asyncio.run(run.main(args, overrides))
    launch.assert_awaited_once_with(args, [
        'trainer.project_name=decomposer-mix-opd', 'trainer.experiment_name=mix-test'],
        prepare_data=mix.prepare)
    assert args.gpus == [4, 5, 6] and args.training_gpus == 2
    assert args.resume_from == Path('/previous/global_step_31')
