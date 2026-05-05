"""
Ingest sign-language images from Hugging Face datasets, local video trees, or local image trees.

Writes into data/<class_folder>/ using a YAML mapping from external gloss/label strings to folder names.

Examples:
  python ML/ingest_external.py --mode local_videos --src datasets_raw/wlasl/videos --mapping my_map.yaml
  python ML/ingest_external.py --mode hf --dataset username/dataset-name --split train --mapping my_map.yaml --max-per-class 100

See ML/download_dataset.md and DATASETS.md for sources and licenses.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data"
VALID_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _norm_key(s: str) -> str:
    t = str(s).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(t.split())


def load_mapping(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise SystemExit("Install PyYAML: pip install pyyaml") from e
        raw = yaml.safe_load(text)
    else:
        import json

        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("Mapping file must be a YAML/JSON object (dict)")
    errors: List[str] = []
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if k is None:
            errors.append("mapping has a null key")
            continue
        if not isinstance(k, str):
            errors.append(f"mapping key must be a string, got {type(k).__name__}")
            continue
        if k.strip().startswith("#"):
            continue
        if v is None:
            errors.append(f"mapping value for {k!r} is null")
            continue
        if not isinstance(v, str):
            errors.append(f"mapping value for {k!r} must be a string, got {type(v).__name__}")
            continue
        target = v.strip()
        if not target:
            errors.append(f"mapping value for {k!r} is empty")
            continue
        target_path = Path(target)
        if target_path.is_absolute() or len(target_path.parts) != 1:
            errors.append(
                f"mapping value for {k!r} must be a single folder name under data/, got {target!r}"
            )
            continue
        if ".." in target_path.parts:
            errors.append(f"mapping value for {k!r} cannot contain '..' (got {target!r})")
            continue
        norm_key = _norm_key(k)
        if norm_key in out and out[norm_key] != target:
            errors.append(
                f"mapping key {k!r} collides after normalization with a different target ({out[norm_key]!r} vs {target!r})"
            )
            continue
        out[norm_key] = target
    if errors:
        detail = "\n".join(f"- {msg}" for msg in errors)
        raise ValueError(f"Invalid mapping file {path}:\n{detail}")
    return out


def map_gloss(raw: str, mapping: Dict[str, str]) -> Optional[str]:
    if raw is None:
        return None
    key = _norm_key(str(raw))
    if key in mapping:
        return mapping[key]
    return None


def _ensure_cv2():
    try:
        import cv2  # noqa: F401

        return cv2
    except ImportError as e:
        raise SystemExit("OpenCV required: pip install opencv-python-headless") from e


def sample_video_frames(video_path: Path, num_frames: int) -> List["Any"]:
    """Return list of BGR numpy frames (OpenCV)."""
    cv2 = _ensure_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return []
    indices: List[int] = []
    if num_frames >= total:
        indices = list(range(total))
    else:
        margin = max(1, int(total * 0.05))
        lo, hi = margin, max(margin + 1, total - margin)
        span = max(1, hi - lo)
        for j in range(num_frames):
            indices.append(lo + int(j * span / max(1, num_frames - 1)) if num_frames > 1 else lo)
    frames: List[Any] = []
    for idx in sorted(set(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def _bgr_to_rgb_save(frame: Any, dest: Path) -> None:
    cv2 = _ensure_cv2()
    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(dest, format="JPEG", quality=92)


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_gloss(row: dict, gloss_field: str) -> Optional[str]:
    if gloss_field and gloss_field != "auto" and gloss_field in row:
        v = row[gloss_field]
        return str(v) if v is not None else None
    for key in ("gloss", "word", "label", "text", "sign", "name", "class"):
        if key in row and row[key] is not None:
            return str(row[key])
    return None


def _pick_video_path(row: dict, video_field: str) -> Optional[Path]:
    if video_field and video_field != "auto" and video_field in row:
        v = row[video_field]
    else:
        v = None
        for key in ("video", "file", "path", "clip"):
            if key in row:
                v = row[key]
                break
    if v is None:
        return None
    if isinstance(v, dict) and "path" in v:
        p = v["path"]
        return Path(p) if p else None
    if isinstance(v, str) and Path(v).suffix.lower() in VIDEO_EXT.union({".mp4"}):
        return Path(v)
    return None


def _pick_pil_image(row: dict, image_field: str):
    if image_field and image_field != "auto" and image_field in row:
        return row[image_field]
    for key in ("image", "img", "pixel_values"):
        if key in row:
            return row[key]
    return None


def ingest_local_videos(
    src: Path,
    mapping: Dict[str, str],
    out_dir: Path,
    max_per_class: int,
    frames_per_video: int,
    dry_run: bool,
    merge: bool,
    seen_hashes: Dict[str, Set[str]],
) -> Tuple[int, int]:
    """Expect src/<gloss_or_mapped>/<videos>."""
    written = 0
    skipped = 0
    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        gloss = sub.name
        folder = map_gloss(gloss, mapping) or map_gloss(_norm_key(gloss), mapping)
        if folder is None:
            folder = map_gloss(gloss.replace("_", " "), mapping)
        if folder is None:
            skipped += 1
            continue
        dest = out_dir / folder
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
        count = len(list(dest.glob("*.jpg"))) + len(list(dest.glob("*.png"))) if dest.exists() else 0
        if not merge and dest.exists() and not dry_run:
            shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            count = 0
        for vid in sorted(sub.iterdir()):
            if vid.suffix.lower() not in VIDEO_EXT:
                continue
            if count >= max_per_class:
                break
            frames = sample_video_frames(vid, frames_per_video)
            if not frames:
                continue
            base = vid.stem
            for i, fr in enumerate(frames):
                if count >= max_per_class:
                    break
                fname = f"{base}_f{i}.jpg"
                tmp = dest / f".tmp_{fname}"
                final = dest / fname
                if dry_run:
                    count += 1
                    written += 1
                    continue
                _bgr_to_rgb_save(fr, tmp)
                digest = _file_md5(tmp)
                bucket = seen_hashes.setdefault(folder, set())
                if digest in bucket:
                    tmp.unlink(missing_ok=True)
                    continue
                bucket.add(digest)
                tmp.replace(final)
                count += 1
                written += 1
    return written, skipped


def ingest_local_images(
    src: Path,
    mapping: Dict[str, str],
    out_dir: Path,
    max_per_class: int,
    dry_run: bool,
    merge: bool,
    seen_hashes: Dict[str, Set[str]],
) -> Tuple[int, int]:
    written = 0
    skipped = 0
    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        folder = map_gloss(sub.name, mapping)
        if folder is None:
            skipped += 1
            continue
        dest = out_dir / folder
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
        imgs = [p for p in sub.iterdir() if p.suffix.lower() in VALID_IMG_EXT]
        count = sum(1 for _ in dest.iterdir()) if dest.exists() else 0
        if not merge and dest.exists() and not dry_run:
            shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            count = 0
        for img_path in sorted(imgs):
            if count >= max_per_class:
                break
            suffix = img_path.suffix.lower() or ".png"
            fname = f"ingested_{img_path.stem}{suffix}"
            final = dest / fname
            if dry_run:
                count += 1
                written += 1
                continue
            digest = _file_md5(img_path)
            bucket = seen_hashes.setdefault(folder, set())
            if digest in bucket:
                continue
            bucket.add(digest)
            shutil.copy2(img_path, final)
            count += 1
            written += 1
    return written, skipped


def ingest_hf(
    dataset_id: str,
    split: str,
    mapping: Dict[str, str],
    out_dir: Path,
    max_per_class: int,
    frames_per_video: int,
    dry_run: bool,
    _merge: bool,
    gloss_field: str,
    video_field: str,
    image_field: str,
    max_rows: Optional[int],
    seen_hashes: Dict[str, Set[str]],
) -> Tuple[int, int]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install datasets: pip install datasets huggingface_hub") from e

    ds = load_dataset(dataset_id, split=split)
    per_class: Dict[str, int] = {}
    written = 0
    skipped = 0

    for idx, row in enumerate(ds):
        if max_rows is not None and idx >= max_rows:
            break
        gloss = _pick_gloss(row, gloss_field)
        if not gloss:
            skipped += 1
            continue
        folder = map_gloss(gloss, mapping)
        if folder is None:
            skipped += 1
            continue
        n = per_class.get(folder, 0)
        if n >= max_per_class:
            continue
        dest = out_dir / folder
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)

        vid_path = _pick_video_path(row, video_field)
        if vid_path is not None and vid_path.is_file():
            frames = sample_video_frames(vid_path, frames_per_video)
            base = vid_path.stem
            for i, fr in enumerate(frames):
                if per_class.get(folder, 0) >= max_per_class:
                    break
                fname = f"hf_{idx}_{base}_f{i}.jpg"
                if dry_run:
                    per_class[folder] = per_class.get(folder, 0) + 1
                    written += 1
                    continue
                tmp = dest / f".tmp_{fname}"
                _bgr_to_rgb_save(fr, tmp)
                digest = _file_md5(tmp)
                bucket = seen_hashes.setdefault(folder, set())
                if digest in bucket:
                    tmp.unlink(missing_ok=True)
                    continue
                bucket.add(digest)
                tmp.replace(dest / fname)
                per_class[folder] = per_class.get(folder, 0) + 1
                written += 1
            continue

        pil_img = _pick_pil_image(row, image_field)
        if pil_img is not None:
            from PIL import Image

            if not isinstance(pil_img, Image.Image):
                skipped += 1
                continue
            if per_class.get(folder, 0) >= max_per_class:
                continue
            fname = f"hf_{idx}.jpg"
            if dry_run:
                per_class[folder] = per_class.get(folder, 0) + 1
                written += 1
                continue
            tmp = dest / f".tmp_{fname}"
            pil_img.convert("RGB").save(tmp, format="JPEG", quality=92)
            digest = _file_md5(tmp)
            bucket = seen_hashes.setdefault(folder, set())
            if digest in bucket:
                tmp.unlink(missing_ok=True)
                continue
            bucket.add(digest)
            tmp.replace(dest / fname)
            per_class[folder] = per_class.get(folder, 0) + 1
            written += 1
            continue

        skipped += 1

    return written, skipped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest external sign-language media into data/<class>/")
    p.add_argument("--mode", choices=("hf", "local_videos", "local_images"), required=True)
    p.add_argument("--mapping", type=Path, required=True, help="YAML or JSON gloss -> folder map")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output root (default: repo data/)")
    p.add_argument("--src", type=Path, help="Local root for local_videos / local_images")
    p.add_argument("--dataset", type=str, help="Hugging Face dataset id for mode=hf")
    p.add_argument("--split", type=str, default="train", help="HF dataset split")
    p.add_argument("--max-per-class", type=int, default=300)
    p.add_argument("--frames-per-video", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-merge", action="store_true", help="Clear each destination class folder before writing")
    p.add_argument("--gloss-field", type=str, default="auto")
    p.add_argument("--video-field", type=str, default="auto")
    p.add_argument("--image-field", type=str, default="auto")
    p.add_argument("--max-rows", type=int, default=None, help="HF only: stop after N rows")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    mapping = load_mapping(args.mapping.resolve())
    out_dir = args.out_dir.resolve()
    merge = not args.no_merge
    seen_hashes: Dict[str, Set[str]] = {}

    if args.dry_run:
        print("[dry-run] no files will be written")

    if args.mode in ("local_videos", "local_images"):
        if not args.src:
            print("--src required for local modes", file=sys.stderr)
            return 2
        src = args.src.resolve()
        if not src.is_dir():
            print(f"Not a directory: {src}", file=sys.stderr)
            return 2
        if args.mode == "local_videos":
            w, s = ingest_local_videos(
                src, mapping, out_dir, args.max_per_class, args.frames_per_video, args.dry_run, merge, seen_hashes
            )
        else:
            w, s = ingest_local_images(src, mapping, out_dir, args.max_per_class, args.dry_run, merge, seen_hashes)
        print(f"Ingested (or dry-run counted): {w} files; unmapped folders skipped: {s}")
        return 0

    if not args.dataset:
        print("--dataset required for mode=hf", file=sys.stderr)
        return 2
    w, s = ingest_hf(
        args.dataset,
        args.split,
        mapping,
        out_dir,
        args.max_per_class,
        args.frames_per_video,
        args.dry_run,
        merge,
        args.gloss_field,
        args.video_field,
        args.image_field,
        args.max_rows,
        seen_hashes,
    )
    print(f"Ingested (or dry-run counted): {w} files; skipped rows: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
