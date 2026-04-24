"""
Upload local ./data/ training images to the Hugging Face dataset repository.

Uses the resumable `hf upload-large-folder` CLI with the `data` directory as LOCAL_PATH
(class folders land at the dataset repo root — the layout expected by pull_data_from_hf.py).

Requires a Hub token with write access (`hf auth login` or HF_TOKEN).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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


def find_hf_cli() -> str:
    win = ROOT / "venv" / "Scripts" / "hf.exe"
    if win.is_file():
        return str(win)
    exe = shutil.which("hf")
    if exe:
        return exe
    raise SystemExit("Could not find `hf` CLI. Install huggingface_hub in ./venv (see requirements.txt).")


def main() -> None:
    p = argparse.ArgumentParser(description="Upload ./data/ to a Hugging Face dataset repo (resumable).")
    p.add_argument("--repo-id", default="", help=f"HF dataset repo id (default: {DEFAULT_HF_DATASET_REPO}).")
    args = p.parse_args()

    repo_id = resolve_repo_id(args.repo_id)
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Missing data directory: {DATA_DIR}")

    hf = find_hf_cli()
    # No --include: avoids Windows glob expansion into thousands of CLI args.
    cmd = [
        hf,
        "upload-large-folder",
        repo_id,
        str(DATA_DIR),
        "--repo-type",
        "dataset",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    print(f"Finished upload to dataset {repo_id} (class folders at repo root)")


if __name__ == "__main__":
    main()
    sys.exit(0)
