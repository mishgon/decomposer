"""Run inside the Gym image, with its subagent directory on PYTHONPATH."""
import unittest
try:
    from graph import validate_transport_numbers
except ModuleNotFoundError:
    validate_transport_numbers = None


@unittest.skipIf(validate_transport_numbers is None, "Run inside the Gym subagent environment")
class TransportNumbersTest(unittest.TestCase):
    def test_limits(self):
        validate_transport_numbers({"values": [-(2**63), 2**64-1, True, "1000000000000000000000"]})
        for number in (10**21, -(2**63)-1, 2**64):
            with self.assertRaisesRegex(ValueError, "64-bit"):
                validate_transport_numbers({"nested": [{"page": number}]})


if __name__ == '__main__':
    unittest.main()
