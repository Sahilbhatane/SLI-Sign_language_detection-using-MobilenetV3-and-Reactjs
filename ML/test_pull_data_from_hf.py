"""Unit tests for ML/pull_data_from_hf.py repo resolution."""

import os
import unittest
from unittest.mock import patch

from pull_data_from_hf import DEFAULT_HF_DATASET_REPO, resolve_repo_id


class TestResolveRepoId(unittest.TestCase):
    def test_explicit_arg_wins(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_repo_id("org/custom"), "org/custom")

    def test_env_override(self):
        with patch.dict(os.environ, {"SLI_HF_DATASET_REPO": "env/repo"}):
            self.assertEqual(resolve_repo_id(""), "env/repo")

    def test_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_repo_id(""), DEFAULT_HF_DATASET_REPO)


if __name__ == "__main__":
    unittest.main()
