"""Run with the training environment: real veRL configuration and masked loss."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip('verl')
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.trainer.distillation.losses import distillation_loss

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    with patch.dict(os.environ, {'OPD_ROOT': str(ROOT), 'OPD_DATA': '/tmp/data',
                                 'OPD_ARTIFACTS': '/tmp/run', 'MODEL_PATH': '/tmp/student',
                                 'WS_WORKER_URL': 'http://unused', 'SYNTH_WORKER_URL': 'http://unused'}), \
            initialize_config_dir(config_dir=str(ROOT / 'opd/wideseek'), version_base=None):
        c = compose(config_name=name, overrides=['hydra.searchpath=[pkg://verl.trainer.config]',
                                                'trainer.experiment_name=config-test'])
        OmegaConf.resolve(c)
        return c


def test_smoke_is_two_updates_with_same_loss_and_optimizer():
    smoke, full = config('smoke'), config('full')
    assert smoke.actor_rollout_ref == full.actor_rollout_ref
    assert smoke.distillation == full.distillation
    assert smoke.rollout.total_rollout_steps == 16
    assert full.wideseek.timeout == 900
    assert full.data.max_prompt_length + full.data.max_response_length == 32768
    assert full.rollout.total_rollout_steps == 512
    assert full.data.shuffle is True
    assert full.data.seed == 42
    assert full.actor_rollout_ref.model.override_config.attn_implementation == "flash_attention_2"
    assert smoke.trainer.val_before_train
    assert smoke.trainer.test_freq == 8
    assert smoke.trainer.save_freq == 1
    assert smoke.distillation.distillation_loss.use_task_rewards is False
    assert smoke.actor_rollout_ref.rollout.n == 1
    assert smoke.actor_rollout_ref.actor.ppo_mini_batch_size == 8
    assert smoke.async_training.require_batches == 1


def test_real_loss_masks_observations_and_produces_gradient():
    c = config('smoke')
    actor = omega_conf_to_dataclass(c.actor_rollout_ref.actor)
    distill = omega_conf_to_dataclass(c.distillation)
    logits = torch.tensor([-1., -1., -1., -1., -1.], requires_grad=True)
    data = {'prompts': torch.tensor([[10, 11]]), 'responses': torch.tensor([[12, 13, 14]]),
            'attention_mask': torch.ones(1, 5, dtype=torch.long), 'response_mask': torch.tensor([[1, 0, 1]]),
            'old_log_probs': torch.tensor([[-1., -1., -1.]]),
            'teacher_logprobs': torch.tensor([[-1.], [-.5], [-50.], [-.5], [0.]]),
            'dp_size': 1, 'batch_num_tokens': 2, 'global_batch_size': 1}
    loss, _ = distillation_loss(actor, distill, {'log_probs': logits}, data)
    loss.backward()
    assert logits.grad[2].item() == 0.
    assert logits.grad[1].item() < 0 and logits.grad[3].item() < 0


def test_teacher_next_token_alignment():
    from opd.manager import HostedTeacherClient, HostedTeacherManager
    from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
    c = config('smoke')
    clients = HostedTeacherManager(c).get_client()
    manager = AsyncTeacherLLMServerManager(c, clients)
    assert set(clients) == set(manager.teacher_model_configs)
    client = HostedTeacherClient('/unused')
    client.tokenizer = object()
    with patch.dict(os.environ, {'OPD_ARTIFACTS': '/tmp/test'}), \
            patch('opd.manager.score', AsyncMock(return_value=[0., -.3, -.7])):
        result = asyncio.run(client.generate('test', [10, 11, 12], {'prompt_logprobs': 0}))
    assert result.extra_fields['prompt_ids'] == [[11], [12], [0]]
    assert result.extra_fields['prompt_logprobs'] == [[-.3], [-.7], [0.]]


@pytest.mark.parametrize('cleanup_error', [False, True])
def test_adapter_uses_shared_gym_and_preserves_tokens(tmp_path, monkeypatch, cleanup_error):
    from opd.wideseek.agent_loop import WideSeekAgentLoop
    monkeypatch.setenv('OPD_DATA', str(tmp_path))
    monkeypatch.setenv('OPD_ARTIFACTS', str(tmp_path / 'run'))
    (tmp_path / 'tasks.jsonl').write_text(json.dumps({'task_id': 'a'}) + '\n')
    loop = object.__new__(WideSeekAgentLoop)
    loop.config = config('smoke')
    loop.rollout_config = loop.config.actor_rollout_ref.rollout
    loop.tokenizer = object()
    tokens = SimpleNamespace(prompt_ids=[1], response_ids=[2, 3, 4], mask=[1, 0, 1],
                             logprobs=[-.1, 0., -.2], extra={}, turns=1)
    result = {'status': 'finished', 'evaluation': {'score': .5},
              'cleanup_errors': ['still active'] if cleanup_error else []}
    with patch('opd.wideseek.agent_loop.PolicyTokens', return_value=tokens), \
            patch('opd.wideseek.agent_loop.episode', new=AsyncMock(return_value=result)) as run:
        if cleanup_error:
            with pytest.raises(RuntimeError, match='cleanup/capture/export failed'):
                asyncio.run(loop.run({}, extra_info={'task_id': 'a'}))
        else:
            output = asyncio.run(loop.run({}, extra_info={'task_id': 'a'}))
            assert output.reward_score == .5
            assert output.response_mask == [1, 0, 1]
            assert run.await_args.kwargs['policy'].tokens is tokens
    trace = json.loads(next((tmp_path / 'run/episodes').glob('*/trajectory.json')).read_text())
    assert trace['response_mask'] == [1, 0, 1]


def test_effective_recipe_matches_final_synth_except_gym():
    wide = config('full')
    with patch.dict(os.environ, {'OPD_ROOT': str(ROOT), 'OPD_DATA': '/tmp/data',
                                 'OPD_ARTIFACTS': '/tmp/run', 'MODEL_PATH': '/tmp/student',
                                 'SYNTH_WORKER_URL': 'http://unused'}), \
            initialize_config_dir(config_dir=str(ROOT / 'opd/synth'), version_base=None):
        synth = compose(config_name='experiment', overrides=[
            'hydra.searchpath=[pkg://verl.trainer.config]', 'data.val_batch_size=32',
            'data.max_response_length=24576',
            'actor_rollout_ref.rollout.val_kwargs.n=1', 'trainer.experiment_name=config-test'])
        OmegaConf.resolve(synth)
    a, b = OmegaConf.to_container(wide), OmegaConf.to_container(synth)
    for section in ('model', 'actor'):
        assert a['actor_rollout_ref'][section]['use_fused_kernels'] is True
        assert b['actor_rollout_ref'][section]['use_fused_kernels'] is True
    assert a['actor_rollout_ref']['model']['enable_gradient_checkpointing'] is True
    a.pop('wideseek'); b.pop('synth')
    for c in (a, b):
        c['trainer'].pop('project_name')
        c['actor_rollout_ref']['rollout']['trace'].pop('project_name')
        c['actor_rollout_ref']['rollout']['agent'].pop('agent_loop_config_path')
        c['actor_rollout_ref']['rollout']['agent'].pop('default_agent_loop')
    assert a == b


def test_seeded_rotation_consumes_pool_before_repeating():
    from verl.trainer.ppo.utils import create_rl_sampler

    data = config('full').data
    tasks = list(range(1000))
    sampler = create_rl_sampler(data, tasks)
    first, second = list(sampler), list(sampler)
    assert len(first) == len(set(first)) == len(tasks)
    assert set(first) == set(second) == set(tasks)
    assert first != second
    replay = create_rl_sampler(data, tasks)
    assert list(replay) == first
    assert list(replay) == second
