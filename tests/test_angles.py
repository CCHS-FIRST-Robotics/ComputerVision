import unittest

import numpy as np

from ccvision.angles import Angles
from ccvision.utils import rad2deg


class TestStringMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.angles = Angles(640, 400, 75)

    def test_angles(self):
        ah, av = self.angles.get_angle(320, 200)
        self.assertEqual(rad2deg(ah), 0.0)
        self.assertEqual(rad2deg(av), 0.0)

        ah, av = self.angles.get_angle(0, 0)
        self.assertAlmostEqual(rad2deg(ah), -37.5, 1, "1 decimal error")
        self.assertAlmostEqual(rad2deg(av), 25.6, 1, "1 decimal error")

        ah, av = self.angles.get_angle(640, 400)
        self.assertAlmostEqual(rad2deg(ah), 37.5, 1, "1 decimal error")
        self.assertAlmostEqual(rad2deg(av), -25.6, 1, "1 decimal error")

        ah, av = self.angles.get_angle(160, 100)
        self.assertAlmostEqual(rad2deg(ah), -20.98, 1, "1 decimal error")
        self.assertAlmostEqual(rad2deg(av), 13.48, 1, "1 decimal error")

    def test_distance(self):
        d = self.angles.get_distance(10, 0.18)
        self.assertAlmostEqual(d, 7.5, None, "3 decimal error", 0.01)


if __name__ == "__main__":
    unittest.main()
