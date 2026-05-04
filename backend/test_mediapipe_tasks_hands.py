"""Unit tests for Tasks Hands compatibility helpers (no model file required)."""

import unittest
from types import SimpleNamespace

from mediapipe_tasks_hands import tasks_result_to_multi_hand_landmarks


class _NL:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class TestTasksResultConversion(unittest.TestCase):
    def test_empty_hand_landmarks(self):
        res = SimpleNamespace(hand_landmarks=[])
        self.assertIsNone(tasks_result_to_multi_hand_landmarks(res))

    def test_one_hand_roundtrip_shape(self):
        hand = [_NL(0.1, 0.2, 0.3) for _ in range(21)]
        res = SimpleNamespace(hand_landmarks=[hand])
        out = tasks_result_to_multi_hand_landmarks(res)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0].landmark), 21)
        self.assertAlmostEqual(out[0].landmark[0].x, 0.1)
        self.assertAlmostEqual(out[0].landmark[0].y, 0.2)
        self.assertAlmostEqual(out[0].landmark[0].z, 0.3)


if __name__ == "__main__":
    unittest.main()
