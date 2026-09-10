import tempfile
import unittest
from pathlib import Path

from training.toolathlon_gym.prepare_data import make_split


class TaskSplitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("alpha", "beta", "gamma", "delta"):
            (self.root / name).mkdir()
            (self.root / name / "task_config.json").write_text("{}")

    def test_repeatable_disjoint_complete(self):
        result = make_split(self.root, 1, 42)
        self.assertEqual(result, make_split(self.root, 1, 42))
        self.assertFalse(set(result["train"]) & set(result["validation"]))
        self.assertEqual(len(result["train"]), 3)
        self.assertEqual(len(result["validation"]), 1)
        self.assertEqual(len(result["task_config_sha256"]), 4)

    def test_no_holdout_by_default_policy(self):
        self.assertEqual(len(make_split(self.root, 0, 42)["train"]), 4)

    def test_reject_empty_training_split(self):
        for count in (-1, 4, 5):
            with self.assertRaises(ValueError):
                make_split(self.root, count, 42)

    def test_reject_missing_config(self):
        (self.root / "alpha/task_config.json").unlink()
        with self.assertRaises(ValueError):
            make_split(self.root, 1, 42)

    def test_config_change_is_visible(self):
        before = make_split(self.root, 1, 42)
        (self.root / "alpha/task_config.json").write_text('{"changed": true}')
        after = make_split(self.root, 1, 42)
        self.assertNotEqual(before["task_config_sha256"], after["task_config_sha256"])
