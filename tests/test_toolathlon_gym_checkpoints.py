import tempfile
import unittest
from pathlib import Path

from training.toolathlon_gym.select_checkpoints import select


class CheckpointSelectionTest(unittest.TestCase):
    def test_best_and_last_can_differ(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (2, 4):
                (root / f"global_step_{step}" / "actor").mkdir(parents=True)
            panels = [{"step": step, "complete": True, "panel": "toolathlon_gym/train_probe",
                       "reward": reward} for step, reward in [(0, 0.9), (2, 0.7), (4, 0.5)]]
            self.assertEqual(select(root, panels), {"last": 4, "best": 2})
            self.assertEqual((root / "best").resolve(), root / "global_step_2")
            self.assertEqual((root / "last").resolve(), root / "global_step_4")
            panels[-1]["reward"] = 0.8
            self.assertEqual(select(root, panels), {"last": 4, "best": 4})

    def test_incomplete_panel_cannot_win(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "global_step_2" / "actor").mkdir(parents=True)
            self.assertEqual(select(root, [{"step": 2, "complete": False}]), {"last": 2})


if __name__ == "__main__":
    unittest.main()
