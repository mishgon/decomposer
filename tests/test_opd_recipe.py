"""CPU checks against the pinned veRL, including the real masked OPD loss."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch, AsyncMock

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.trainer.distillation.losses import distillation_loss

ROOT = Path(__file__).resolve().parents[1]


class RecipeTests(unittest.TestCase):
    def config(self, name):
        env = {'RL_ROOT': str(ROOT), 'RL_DATA': '/tmp/data', 'RL_ARTIFACTS': '/tmp/run',
               'MODEL_PATH': '/tmp/student', 'OPD_TEACHER_MODEL': 'teacher'}
        with patch.dict(os.environ, env), initialize_config_dir(
                config_dir=str(ROOT / 'training/opd/toolathlon_gym'), version_base=None):
            config = compose(config_name=name, overrides=['hydra.searchpath=[pkg://verl.trainer.config]',
                                                          'trainer.experiment_name=recipe-test'])
            OmegaConf.resolve(config)
            return config

    def test_smoke_full_share_optimizer(self):
        a, b = self.config('smoke'), self.config('full')
        a.trainer.experiment_name = b.trainer.experiment_name
        self.assertEqual(OmegaConf.to_container(a), OmegaConf.to_container(b))
        self.assertEqual(a.actor_rollout_ref.actor.optim.lr, 1.5e-5)
        self.assertEqual(a.actor_rollout_ref.model.lora_rank, 32)
        self.assertEqual(a.actor_rollout_ref.rollout.n, 1)
        d = omega_conf_to_dataclass(a.distillation)
        self.assertFalse(d.distillation_loss.use_task_rewards)
        self.assertTrue(d.distillation_loss.use_policy_gradient)
        self.assertEqual(d.n_gpus_per_node * d.nnodes, 0)

    def test_real_loss_masks_observations_and_produces_gradient(self):
        c = self.config('smoke')
        actor = omega_conf_to_dataclass(c.actor_rollout_ref.actor)
        distill = omega_conf_to_dataclass(c.distillation)
        # Two prompt positions, then student/tool/student positions.
        logits = torch.tensor([-1., -1., -1., -1., -1.], requires_grad=True)
        data = {'prompts': torch.tensor([[10, 11]]), 'responses': torch.tensor([[12, 13, 14]]),
                'attention_mask': torch.ones(1, 5, dtype=torch.long), 'response_mask': torch.tensor([[1, 0, 1]]),
                'old_log_probs': torch.tensor([[-1., -1., -1.]]),
                'teacher_logprobs': torch.tensor([[-1.], [-.5], [-50.], [-.5], [0.]]),
                'dp_size': 1, 'batch_num_tokens': 2, 'global_batch_size': 1}
        loss, _ = distillation_loss(actor, distill, {'log_probs': logits}, data)
        loss.backward()
        self.assertEqual(logits.grad[2].item(), 0.)
        self.assertLess(logits.grad[1].item(), 0.)
        self.assertLess(logits.grad[3].item(), 0.)

    def test_teacher_manager_routing_matches_verl(self):
        from training.opd.toolathlon_gym.manager import HostedTeacherManager
        from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
        config = self.config('smoke')
        clients = HostedTeacherManager(config).get_client()
        manager = AsyncTeacherLLMServerManager(config, clients)
        self.assertEqual(set(clients), set(manager.teacher_model_configs))


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_teacher_alignment_matches_verl_next_token_convention(self):
        from training.opd.toolathlon_gym.manager import HostedTeacherClient
        client = HostedTeacherClient('/unused')
        client.tokenizer = object()
        with patch.dict(os.environ, {'RL_ARTIFACTS': '/tmp/opd-test'}), patch(
                'training.opd.toolathlon_gym.manager.score', AsyncMock(return_value=[0., -.3, -.7])):
            result = await client.generate('test', [10, 11, 12], {'prompt_logprobs': 0})
        self.assertEqual(result.extra_fields['prompt_ids'], [[11], [12], [0]])
        self.assertEqual(result.extra_fields['prompt_logprobs'], [[-.3], [-.7], [0.]])
