"""
Download training images from the project's Hugging Face dataset repo into ./data/.

Training scripts already read from data/ on disk; this script is the supported way to
populate that folder on a fresh clone without committing images to git.

The Hub layout matches `hf upload-large-folder <repo> data` (class folders live at the
dataset repo root: agree/, again/, …).

Environment:
  SLI_HF_DATASET_REPO — optional override (default: canonical repo below).
  HF_TOKEN / HUGGING_FACE_HUB_TOKEN — only required for private datasets or CI without cached login.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DEFAULT_HF_DATASET_REPO = "SahilBhatane/sli"


def resolve_repo_id(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("SLI_HF_DATASET_REPO", "").strip()
    if env:
        return env
    return DEFAULT_HF_DATASET_REPO


def main() -> None:
    p = argparse.ArgumentParser(description="Sync ./data/ from a Hugging Face dataset repository.")
    p.add_argument(
        "--repo-id",
        default="",
        help=f"HF dataset repo id (default: {DEFAULT_HF_DATASET_REPO} or SLI_HF_DATASET_REPO).",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit hash on the Hub.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved repo id and exit without downloading.",
    )
    args = p.parse_args()

    repo_id = resolve_repo_id(args.repo_id)
    if args.dry_run:
        print(f"Would download dataset {repo_id!r} into {DATA_DIR}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from e

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Class folders at dataset repo root (from `hf upload-large-folder repo ./data`).
    patterns = [
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.bmp",
        "**/*.webp",
    ]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(DATA_DIR),
        allow_patterns=patterns,
    )
    print(f"Synced images from {repo_id} into {DATA_DIR}")


if __name__ == "__main__":
    main()
    sys.exit(0)
