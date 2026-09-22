"""CPU-only recipe composition checks; run with the installed veRL environment."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "training/rl/toolathlon_gym"


class RecipeTests(unittest.TestCase):
    def test_shared_recipe_and_schedule(self):
        env = {"RL_ROOT": str(ROOT), "RL_DATA": "/tmp/data",
               "RL_ARTIFACTS": "/tmp/run", "MODEL_PATH": "/tmp/model"}
        with patch.dict(os.environ, env), initialize_config_dir(
                config_dir=str(CONFIGS), version_base=None):
            overrides = ["hydra.searchpath=[pkg://verl.trainer.config]"]
            full = compose(config_name="full", overrides=overrides)
            smoke = compose(config_name="smoke", overrides=overrides)
            cold = compose(config_name="cold-start", overrides=overrides)
            for config in (full, smoke, cold):
                self.assertEqual(config.trainer.save_freq, 1)
                self.assertEqual(config.trainer.max_actor_ckpt_to_keep, -1)
                self.assertEqual(list(config.actor_rollout_ref.actor.checkpoint.save_contents),
                                 ["model", "optimizer", "extra"])
                self.assertEqual(config.actor_rollout_ref.model.lora_rank, 32)
                self.assertEqual(config.actor_rollout_ref.model.lora_alpha, 64)
                self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 1.5e-5)
                self.assertFalse(config.algorithm.norm_adv_by_std_in_grpo)
                self.assertEqual(config.actor_rollout_ref.rollout.n, 10)
                self.assertEqual(config.actor_rollout_ref.rollout.val_kwargs.n, 8)
                self.assertIsNone(config.trainer.total_training_steps)
                self.assertIsNone(config.rollout.total_rollout_steps)
                self.assertEqual(config.trainer.total_epochs, 1)
                self.assertEqual(config.trainer.test_freq, 8)
                self.assertEqual(config.data.train_batch_size, 0)
                self.assertEqual(config.actor_rollout_ref.actor.ppo_mini_batch_size, 1)
                self.assertEqual(config.actor_rollout_ref.actor.ppo_epochs, 2)
                self.assertTrue(config.actor_rollout_ref.rollout.multi_turn.enable)
                self.assertTrue(config.actor_rollout_ref.model.lora.merge)
                self.assertTrue(config.algorithm.rollout_correction.bypass_mode)
                self.assertFalse(config.actor_rollout_ref.hybrid_engine)
                self.assertTrue(config.async_training.partial_rollout)
                self.assertEqual(config.rollout.n * config.async_training.concurrent_samples_per_replica, 20)
            recipes = []
            for config in (full, smoke, cold):
                config.trainer.experiment_name = "comparison"
                recipe = OmegaConf.to_container(config, resolve=True)
                recipes.append(recipe)
            self.assertEqual(recipes[0], recipes[1])
            self.assertEqual(recipes[0], recipes[2])

    def test_task_pools(self):
        pools = {name: json.loads((CONFIGS / "task_pools" / f"{name}.json").read_text())["tasks"]
                 for name in ("full", "cold-start", "smoke")}
        self.assertEqual([len(pools[n]) for n in pools], [346, 191, 2])
        full_ids = {r["task_id"] for r in pools["full"]}
        for name, rows in pools.items():
            if name == "smoke":
                self.assertTrue(all(max(r["scores"]) == 1 and sum(r["scores"]) == 1 for r in rows))
                continue  # Explicit binary smoke pool is separate from partial-reward full.
            self.assertTrue({r["task_id"] for r in rows} <= full_ids)
            for row in rows:
                self.assertTrue(any(0 < s < 1 for s in row["scores"]))
                self.assertFalse(all(s >= .9 for s in row["scores"]))
                if name == "cold-start":
                    self.assertLessEqual(row["mean_minutes"], 30)
                    self.assertGreater(row["score_range"], .1)
        self.assertTrue(any(max(r["scores"]) < .9 for r in pools["full"]))


if __name__ == "__main__":
    unittest.main()
