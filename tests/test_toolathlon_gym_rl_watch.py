import unittest

from training.toolathlon_gym.watch import duration, estimate_eta, planned_evaluations, trend


class WatchTests(unittest.TestCase):
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
