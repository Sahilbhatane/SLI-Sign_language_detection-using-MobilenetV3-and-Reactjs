"""Unit tests for LLM correction safeguards (run: python -m unittest test_llm_sanitize from backend/)."""

import unittest

from extra_routes import _sanitize_llm_correction, _token_jaccard


class TestLlmSanitize(unittest.TestCase):
    def test_token_jaccard_overlap(self) -> None:
        self.assertAlmostEqual(_token_jaccard("hello world", "hello there world"), 2 / 3, delta=0.01)

    def test_rejects_unrelated_rewrite(self) -> None:
        out = _sanitize_llm_correction("I go store", "The capital of France is Paris.", 128)
        self.assertEqual(out, "I go store")

    def test_accepts_minor_edit(self) -> None:
        out = _sanitize_llm_correction("I am fine", "I am fine today", 128)
        self.assertEqual(out, "I am fine today")


if __name__ == "__main__":
    unittest.main()
