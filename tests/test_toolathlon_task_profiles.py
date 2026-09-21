import unittest

from training.toolathlon_gym.task_profiles import select_profiles
from training.toolathlon_gym.watch import run_plan


class ProfileTests(unittest.TestCase):
    def test_selection_boundaries_and_zero_successes(self):
        cases = {"zero-pass": ([0, .4, .5, .6, .8], 30),
                 "easy": ([.95] * 5, 5), "binary": ([0, 1, 0, 1, 0], 5),
                 "small-range": ([0, .1, 0, .1, 0], 5),
                 "slow": ([0, .5, 0, .5, 0], 30.01)}
        manifest = {"config": {"repetitions": 5}, "tasks": list(cases), "episodes": {}}
        for name, (scores, minutes) in cases.items():
            for i, score in enumerate(scores, 1):
                manifest["episodes"][f"{name}/rep-{i:03d}"] = {
                    "status": "completed", "partial_score": score, "elapsed_seconds": minutes * 60}
        pools = select_profiles(manifest, ["zero-pass"])
        self.assertEqual({r["task_id"] for r in pools["full"]}, {"zero-pass", "small-range", "slow"})
        self.assertEqual([r["task_id"] for r in pools["cold-start"]], ["zero-pass"])
        binary_smoke = select_profiles(manifest, ["binary"])
        self.assertEqual([r["task_id"] for r in binary_smoke["smoke"]], ["binary"])
        self.assertEqual(binary_smoke["full"], pools["full"])

    def test_unbounded_group_plan(self):
        config = {"trainer": {"total_epochs": 1, "test_freq": 8, "val_before_train": True},
                  "data": {}, "rollout": {"total_rollout_steps": None},
                  "async_training": {"require_batches": 1, "trigger_parameter_sync_step": 1},
                  "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 1},
                                        "rollout": {"n": 10, "val_kwargs": {"n": 8}}}}
        self.assertEqual(run_plan(config, 20, 2), (20, {0, 8, 16}, 16, 248))


if __name__ == "__main__":
    unittest.main()
