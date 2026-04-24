"""
Download and unpack Indian Sign Language phrase datasets.

Sources (CC BY 4.0):
  - w7fgy7jvs8 v2: 44 ISL phrase classes, 40 PNG images each (680x480). Image dataset.
  - y8vg69brn2:    41 ISL phrase classes, MediaPipe .npy landmark arrays (NOT images).

The helper calls Mendeley's public metadata API
(``/api/datasets/<id>/files?version=<ver>``), collects presigned download URLs from
``content_details.download_url``, and streams each asset to ``datasets_raw/<key>/``.
On any failure it prints the browser URL so you can download manually.

Next step (phrases_v2):
  python ML/gen_identity_mapping.py "datasets_raw/phrases_v2/images for phrases" \\
      datasets_raw/phrases_v2_mapping.yaml
  python ML/ingest_external.py --mode local_images \\
      --src "datasets_raw/phrases_v2/images for phrases" \\
      --mapping datasets_raw/phrases_v2_mapping.yaml \\
      --max-per-class 400
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "datasets_raw"
USER_AGENT = "SLI-dataset-downloader/1.0"
METADATA_URL = "https://data.mendeley.com/api/datasets/{id}/files?version={version}"

MENDELEY_DATASETS: Dict[str, Dict[str, str]] = {
    "phrases_v2": {
        "id": "w7fgy7jvs8",
        "version": "2",
        "name": "ISL phrases v2 (44 classes, PNG images)",
        "page": "https://data.mendeley.com/datasets/w7fgy7jvs8/2",
        "license": "CC BY 4.0",
        "is_image_dataset": "1",
    },
    "common_phrases": {
        "id": "y8vg69brn2",
        "version": "1",
        "name": "ISL common phrases (41 classes, MediaPipe .npy landmarks)",
        "page": "https://data.mendeley.com/datasets/y8vg69brn2/1",
        "license": "CC BY 4.0",
        "is_image_dataset": "0",
    },
}


def list_files(ds: Dict[str, str]) -> List[Dict]:
    """Query Mendeley's dataset-files endpoint for filenames + presigned URLs."""
    url = METADATA_URL.format(id=ds["id"], version=ds["version"])
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=30) as resp:  # noqa: S310 — https only
        if resp.status != 200:
            raise RuntimeError(f"metadata request failed: HTTP {resp.status}")
        payload = json.loads(resp.read().decode("utf-8"))
    entries: List[Dict] = []
    for f in payload:
        filename = f.get("filename")
        dl = (f.get("content_details") or {}).get("download_url")
        if filename and dl:
            entries.append({"filename": filename, "url": dl})
    return entries


def _download(url: str, dest: Path, chunk: int = 65536) -> bool:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urlopen(req, timeout=120) as resp:  # noqa: S310 — https only
            ctype = resp.headers.get("Content-Type", "")
            if resp.status != 200 or "text/html" in ctype.lower():
                return False
            total = int(resp.headers.get("Content-Length") or 0)
            written = 0
            with dest.open("wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    written += len(buf)
                    if total:
                        pct = written * 100.0 / total
                        sys.stdout.write(f"\r  downloading... {written/1e6:.1f} MB / {total/1e6:.1f} MB ({pct:.1f}%)")
                        sys.stdout.flush()
            if total:
                sys.stdout.write("\n")
            return written > 1_000
    except OSError:
        # OSError covers URLError, HTTPError, TimeoutError, and socket errors.
        return False


def _try_unzip(archive: Path, out_dir: Path) -> bool:
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        return True
    except zipfile.BadZipFile:
        return False


def _manual_instructions(ds: Dict[str, str], out_dir: Path) -> None:
    print("\nAutomatic download was not possible.")
    print(f"Open {ds['page']} in a browser and click 'Download All'.")
    print(f"Then unpack the archive into: {out_dir}")
    print(f"The dataset is licensed under {ds['license']}; please retain attribution.")


def download_mendeley(key: str, out_root: Path) -> Optional[Path]:
    ds = MENDELEY_DATASETS.get(key)
    if ds is None:
        raise SystemExit(f"Unknown dataset key: {key}")
    out_dir = out_root / key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== {ds['name']} ==")
    print(f"Target: {out_dir}")
    if ds.get("is_image_dataset") == "0":
        print("NOTE: This dataset contains .npy MediaPipe landmark arrays, not images.")
        print("      It is NOT usable with the EfficientNetV2 image classifier in this repo.")
    if any(out_dir.iterdir()):
        print("Destination already contains files; leaving as-is. Delete it to re-download.")
        return out_dir

    try:
        files = list_files(ds)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Metadata fetch failed: {exc}")
        files = []

    if not files:
        _manual_instructions(ds, out_dir)
        return None

    print(f"Found {len(files)} file(s). License: {ds['license']}. Please retain attribution.")
    ok_any = False
    for entry in files:
        archive = out_dir / entry["filename"]
        if archive.exists() and archive.stat().st_size > 1_000:
            print(f"  already present: {archive.name}")
            ok_any = True
            continue
        print(f"  -> {archive.name}")
        if not _download(entry["url"], archive):
            print(f"  failed: {entry['filename']}")
            archive.unlink(missing_ok=True)
            continue
        ok_any = True
        if archive.suffix.lower() == ".zip":
            if _try_unzip(archive, out_dir):
                archive.unlink(missing_ok=True)
                print(f"  unpacked -> {out_dir}")
            else:
                print(f"  kept archive (could not unzip): {archive}")

    if not ok_any:
        _manual_instructions(ds, out_dir)
        return None
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        choices=sorted(MENDELEY_DATASETS.keys()) + ["all"],
        required=True,
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    keys = list(MENDELEY_DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    any_ok = False
    for key in keys:
        result = download_mendeley(key, out_root)
        any_ok = any_ok or (result is not None)
    if not any_ok:
        print("\nNo dataset downloaded automatically. See printed manual steps above.", file=sys.stderr)
        return 1
    print("\nNext: inspect the extracted folder layout, then merge into data/ via:")
    print("  python ML/ingest_external.py --mode local_images --src <extracted path>\\<class_root> \\")
    print("    --mapping ML/external_gloss_mapping.example.yaml --max-per-class 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
