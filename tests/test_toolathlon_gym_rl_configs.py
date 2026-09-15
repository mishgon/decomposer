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
                self.assertEqual(config.actor_rollout_ref.model.lora_rank, 32)
                self.assertEqual(config.actor_rollout_ref.model.lora_alpha, 64)
                self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 1.5e-5)
                self.assertFalse(config.algorithm.norm_adv_by_std_in_grpo)
                self.assertEqual(config.actor_rollout_ref.rollout.n, 5)
                self.assertEqual(config.actor_rollout_ref.rollout.val_kwargs.n, 5)
            self.assertIsNone(full.trainer.total_training_steps)
            self.assertEqual(overfit.trainer.total_training_steps, 16)
            self.assertEqual(full.trainer.total_epochs, 1)
            self.assertEqual(overfit.trainer.total_epochs, 16)
            overfit.trainer.total_training_steps = full.trainer.total_training_steps
            overfit.trainer.total_epochs = full.trainer.total_epochs
            overfit.trainer.experiment_name = full.trainer.experiment_name
            self.assertEqual(OmegaConf.to_container(full), OmegaConf.to_container(overfit))

    def test_task_pools(self):
        full = json.loads((CONFIGS / "rl_task_pool.json").read_text())["tasks"]
        overfit = json.loads((CONFIGS / "overfit_partial8_pool.json").read_text())["tasks"]
        self.assertEqual(len(full), 194)
        self.assertEqual(len(overfit), 8)
        self.assertTrue({t["task_id"] for t in overfit} <= {t["task_id"] for t in full})


if __name__ == "__main__":
    unittest.main()
