"""CPU-only recipe composition checks; run with the installed veRL environment."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "training/toolathlon_gym"


class RecipeTests(unittest.TestCase):
    def test_shared_recipe_and_schedule(self):
        env = {"RL_ROOT": str(ROOT), "RL_DATA": "/tmp/data",
               "RL_ARTIFACTS": "/tmp/run", "MODEL_PATH": "/tmp/model"}
        with patch.dict(os.environ, env), initialize_config_dir(
                config_dir=str(CONFIGS), version_base=None):
            overrides = ["hydra.searchpath=[pkg://verl.trainer.config]"]
            full = compose(config_name="full", overrides=overrides)
            overfit = compose(config_name="overfit", overrides=overrides)
            for config in (full, overfit):
                self.assertEqual(config.trainer.save_freq, 1)
                self.assertEqual(config.trainer.max_actor_ckpt_to_keep, -1)
                self.assertEqual(list(config.actor_rollout_ref.actor.checkpoint.save_contents),
                                 ["model", "optimizer", "extra"])
                self.assertEqual(config.actor_rollout_ref.model.lora_rank, 32)
                self.assertEqual(config.actor_rollout_ref.model.lora_alpha, 64)
                self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 1.5e-5)
                self.assertFalse(config.algorithm.norm_adv_by_std_in_grpo)
                self.assertEqual(config.actor_rollout_ref.rollout.n, 5)
                self.assertEqual(config.actor_rollout_ref.rollout.val_kwargs.n, 5)
            self.assertIsNone(full.trainer.total_training_steps)
            self.assertEqual(overfit.trainer.total_training_steps, 4)
            self.assertEqual(full.trainer.total_epochs, 1)
            self.assertEqual(overfit.trainer.total_epochs, 4)
            self.assertEqual(full.trainer.test_freq, 8)
            self.assertEqual(overfit.trainer.test_freq, 1)
            self.assertEqual(overfit.data.train_batch_size, 2)
            self.assertEqual(overfit.actor_rollout_ref.actor.ppo_mini_batch_size, 1)
            self.assertEqual(overfit.actor_rollout_ref.actor.ppo_epochs, 2)
            self.assertIn("clearml", overfit.trainer.logger)
            self.assertEqual(full.data.train_batch_size, 8)
            self.assertEqual(full.actor_rollout_ref.actor.ppo_epochs, 1)

    def test_task_pools(self):
        full = json.loads((CONFIGS / "rl_task_pool.json").read_text())["tasks"]
        overfit = json.loads((CONFIGS / "overfit_partial2_pool.json").read_text())["tasks"]
        self.assertEqual(len(full), 194)
        self.assertEqual(len(overfit), 2)
        self.assertTrue({t["task_id"] for t in overfit} <= {t["task_id"] for t in full})


if __name__ == "__main__":
    unittest.main()
