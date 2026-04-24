"""Unit tests for dataset counting and class weight helpers in train_mobilenet.py."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class TestDatasetHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import train_mobilenet as tm  # noqa: PLC0415 — after sys.path

        cls.tm = tm

    def test_count_images_fixture(self):
        tm = self.tm
        fixture = tm.Config.ROOT_DIR / "ML" / "fixtures" / "minimal_dataset"
        self.assertTrue(fixture.is_dir(), "minimal_dataset fixture missing")
        names = sorted(p.name for p in fixture.iterdir() if p.is_dir())
        counts = tm.count_images_per_class(str(fixture), names)
        self.assertEqual(len(counts), len(names))
        self.assertEqual(sum(counts), 8)

    def test_class_weight_none_when_balanced(self):
        tm = self.tm
        self.assertIsNone(tm.compute_class_weight_if_imbalanced([10, 10, 10]))

    def test_class_weight_when_imbalanced(self):
        tm = self.tm
        cw = tm.compute_class_weight_if_imbalanced([1, 3, 10], ratio_threshold=2.0)
        self.assertIsNotNone(cw)
        self.assertEqual(len(cw), 3)
        self.assertGreater(cw[0], cw[2])


if __name__ == "__main__":
    unittest.main()
