"""Unit tests for MediaPipe hand helpers and no-hand behavior (no full ONNX load)."""
import unittest
import numpy as np

from ensemble_inference import (
    hand_bbox_from_landmarks,
    wrist_relative_hand_landmarks,
    HAND_BBOX_PAD_FRAC,
)


class _LM:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def _make_hand(center_x=0.5, center_y=0.5, size=0.1):
    """21 fake landmarks: square-ish spread around center (MediaPipe order not required for bbox)."""
    lms = []
    for i in range(21):
        dx = (i % 7) / 60.0
        dy = (i // 7) / 60.0
        lms.append(_LM(center_x - size + dx, center_y - size + dy, 0.01 * i))
    return lms


class TestHandBbox(unittest.TestCase):
    def test_bbox_padding_clamped_to_frame(self):
        lms = _make_hand(0.5, 0.5, 0.05)
        W, H = 640, 480
        x1, y1, x2, y2 = hand_bbox_from_landmarks(lms, W, H, pad_frac=HAND_BBOX_PAD_FRAC)
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, W)
        self.assertLessEqual(y2, H)

    def test_empty_returns_none(self):
        self.assertIsNone(hand_bbox_from_landmarks([], 100, 100))


class TestWristRelative(unittest.TestCase):
    def test_wrist_is_zero(self):
        lms = _make_hand()
        rel = wrist_relative_hand_landmarks(lms)
        self.assertEqual(rel.shape, (21, 3))
        self.assertTrue(np.allclose(rel[0], 0.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
