import unittest

from training.toolathlon_gym.report import summarize


def episode(task, reward, step=0, source="validation"):
    return ({"task_id": task, "data_source": source,
             "weight_versions": {"min_global_steps": step, "max_global_steps": step}},
            {"reward": reward, "pass": reward == 1})


class RewardReportTests(unittest.TestCase):
    def test_incomplete_panel_has_no_comparison(self):
        report = summarize([episode("a", 1)], {"validation": {"a", "b"}}, 1)
        self.assertFalse(report[0]["complete"])
        self.assertIsNone(report[0]["reward"])
        self.assertIsNone(report[0]["reward_delta"])

    def test_same_task_comparison_ignores_optimizer_rollouts(self):
        rows = [episode("a", .2), episode("b", .4), episode("a", .4, 4),
                episode("b", .6, 4), episode("a", 1, 4, "train")]
        report = summarize(rows, {"validation": {"a", "b"}}, 1)
        self.assertTrue(report[1]["complete"])
        self.assertAlmostEqual(report[1]["reward_delta"], .2)

    def test_duplicate_attempts_are_not_silently_selected(self):
        report = summarize([episode("a", 0), episode("a", 1)], {"validation": {"a"}}, 1)
        self.assertFalse(report[0]["complete"])

    def test_mixed_policy_versions_rejected(self):
        row = episode("a", 1)
        row[0]["weight_versions"]["max_global_steps"] = 1
        with self.assertRaises(ValueError):
            summarize([row], {"validation": {"a"}}, 1)
