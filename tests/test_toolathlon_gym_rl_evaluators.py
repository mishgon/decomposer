import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.toolathlon_gym.check_evaluators import main


class EvaluatorPreflightTests(unittest.TestCase):
    def test_records_infrastructure_failure_without_turning_it_into_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            split.write_text(json.dumps({"train": ["a"], "validation": ["b"]}))
            good, broken = MagicMock(), MagicMock()
            good.score.return_value = {"reward": 0.0}
            broken.start.side_effect = RuntimeError("database unavailable")
            argv = ["check", "--split", str(split), "--output", str(root / "out"),
                    "--workers", "1"]
            with patch("sys.argv", argv), patch(
                "tests.toolathlon_gym.check_evaluators.Episode", side_effect=[good, broken]
            ), self.assertRaises(SystemExit):
                main()
            results = json.loads((root / "out/results.json").read_text())
            self.assertEqual(results[0], {"task": "a", "reward": 0.0})
            self.assertIn("database unavailable", results[1]["error"])
            self.assertNotIn("reward", results[1])
            good.close.assert_called_once()
            broken.close.assert_not_called()  # Episode.start already cleans failed startup.
