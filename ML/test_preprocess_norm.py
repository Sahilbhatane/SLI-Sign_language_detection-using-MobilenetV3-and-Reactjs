"""Training and inference should use the same MobileNet-style normalization."""
import unittest
import numpy as np


def mobilenet_normalize_uint8(rgb_uint8: np.ndarray) -> np.ndarray:
    x = rgb_uint8.astype(np.float32)
    return x / 127.5 - 1.0


class TestMobilenetNormalize(unittest.TestCase):
    def test_mid_gray_near_zero(self):
        # 127.5 maps exactly to 0 under x/127.5 - 1 (same as training pipeline on float images)
        x = np.full((2, 2, 3), 127.5, dtype=np.float32)
        out = x / 127.5 - 1.0
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertTrue(np.allclose(out, 0.0, atol=1e-5))

    def test_white_is_one(self):
        white = np.full((1, 1, 3), 255, dtype=np.uint8)
        out = mobilenet_normalize_uint8(white)
        self.assertTrue(np.allclose(out, 1.0, atol=1e-5))

    def test_black_is_minus_one(self):
        black = np.zeros((1, 1, 3), dtype=np.uint8)
        out = mobilenet_normalize_uint8(black)
        self.assertTrue(np.allclose(out, -1.0, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
