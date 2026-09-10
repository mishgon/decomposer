import unittest

from training.toolathlon_gym.watch import (duration, estimate_eta, planned_evaluations, trend,
                                          manifest_files, run_plan, task_coverage)


class WatchTests(unittest.TestCase):
    def config(self, steps=1, batch=1, n=2, before=False):
        return {"trainer": {"total_training_steps": steps, "total_epochs": 2,
                            "test_freq": 1, "val_before_train": before},
                "data": {"train_batch_size": batch},
                "actor_rollout_ref": {"rollout": {"n": n, "val_kwargs": {"n": 2}}}}

    def test_four_episode_diagnostic(self):
        self.assertEqual(run_plan(self.config(), 1, 1), (1, {1}, 2, 4))

    def test_hundred_task_epoch_run(self):
        config = self.config(steps=None, batch=10, n=8, before=True)
        config["trainer"]["test_freq"] = 10
        self.assertEqual(run_plan(config, 100, 20), (20, {0, 10, 20}, 40, 1720))

    def test_no_validation(self):
        config = self.config(steps=10)
        config["trainer"]["test_freq"] = -1
        self.assertEqual(run_plan(config, 503, 0), (10, set(), 0, 20))

    def test_overlap_is_derived_from_tasks_not_run_name(self):
        text = task_coverage({"train": {"a", "b"}}, {"anything": {"b", "c"}})
        self.assertIn("1 overlap, 1 held-out", text)
        self.assertIn("0 held-out", task_coverage({"x": {"a"}}, {"y": {"a"}}))

    def test_explicit_and_environment_dataset_paths(self):
        self.assertEqual(manifest_files("${oc.env:RL_DATA}/train.parquet", "/data"),
                         ("/data/train.parquet",))
        self.assertEqual(manifest_files(["/a.parquet", "/b.parquet"], ""),
                         ("/a.parquet", "/b.parquet"))
        with self.assertRaises(ValueError):
            manifest_files("${unknown}/train.parquet", "")

    def test_final_validation_counted_once(self):
        self.assertEqual(planned_evaluations(8, 4, True), {0, 4, 8})
        self.assertEqual(planned_evaluations(9, 4, True), {0, 4, 8, 9})
        self.assertEqual(planned_evaluations(1, -1, False), set())

    def test_warmup_eta_is_explicitly_low_confidence(self):
        seconds, label = estimate_eta(2400, 24, 172, 8, [], 3, [])
        self.assertEqual(seconds, 14800)
        self.assertIn("LOW confidence", label)

    def test_measured_eta_includes_validation(self):
        seconds, label = estimate_eta(2400, 44, 172, 7, [600, 800], 2, [1800])
        self.assertEqual(seconds, 7 * 700 + 2 * 1800)
        self.assertIn("measured", label)

    def test_no_fake_eta_before_samples(self):
        self.assertIsNone(estimate_eta(60, 0, 172, 8, [], 3, [])[0])

    def test_trend_handles_constant_and_nonfinite_values(self):
        self.assertEqual(trend([1, 1]), "▄▄")
        self.assertEqual(trend([float("nan")]), "--")
        self.assertEqual(trend([0, 1]), "▁█")
        self.assertEqual(duration(3600), "1h 00m")
