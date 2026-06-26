"""Unit tests for MediaPipe hand helpers and no-hand behavior (no full ONNX load)."""
import unittest
import numpy as np

from ensemble_inference import (
    hand_bbox_from_landmarks,
    hand_bbox_union,
    mirror_overlay_hands_norm_x,
    overlay_hands_from_landmark_results,
    wrist_relative_hand_landmarks,
    HAND_BBOX_PAD_FRAC,
)


class _LM:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _HandLm:
    def __init__(self, landmark):
        self.landmark = landmark


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

    def test_union_encloses_both_boxes(self):
        lms_a = _make_hand(0.25, 0.5, 0.05)
        lms_b = _make_hand(0.75, 0.5, 0.05)
        W, H = 640, 480
        a = hand_bbox_from_landmarks(lms_a, W, H, pad_frac=HAND_BBOX_PAD_FRAC)
        b = hand_bbox_from_landmarks(lms_b, W, H, pad_frac=HAND_BBOX_PAD_FRAC)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        u = hand_bbox_union([a, b])
        self.assertIsNotNone(u)
        self.assertLessEqual(u[0], min(a[0], b[0]))
        self.assertLessEqual(u[1], min(a[1], b[1]))
        self.assertGreaterEqual(u[2], max(a[2], b[2]))
        self.assertGreaterEqual(u[3], max(a[3], b[3]))


class TestOverlayHelpers(unittest.TestCase):
    def test_overlay_from_two_hands(self):
        lms_a = _make_hand(0.3, 0.5, 0.05)
        lms_b = _make_hand(0.7, 0.5, 0.05)
        hands = [_HandLm(landmark=lms_a), _HandLm(landmark=lms_b)]
        out = overlay_hands_from_landmark_results(hands, 640, 480)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]["landmarks_norm"]), 21)

    def test_mirror_flips_x(self):
        hands = [{"bbox_norm": [0.2, 0.1, 0.4, 0.5], "landmarks_norm": [[0.3, 0.4]]}]
        mirrored = mirror_overlay_hands_norm_x(hands)
        self.assertAlmostEqual(mirrored[0]["bbox_norm"][0], 0.6)
        self.assertAlmostEqual(mirrored[0]["landmarks_norm"][0][0], 0.7)


class TestWristRelative(unittest.TestCase):
    def test_wrist_is_zero(self):
        lms = _make_hand()
        rel = wrist_relative_hand_landmarks(lms)
        self.assertEqual(rel.shape, (21, 3))
        self.assertTrue(np.allclose(rel[0], 0.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
