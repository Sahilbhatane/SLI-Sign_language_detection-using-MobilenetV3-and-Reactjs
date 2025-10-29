"""
Preprocess dataset using MediaPipe Hands:
- Iterate through source dataset ../data/class_name folders
- Detect hands and crop per-hand regions, resized to 224x224
- Save crops under ../data_preprocessed/class_name with the same base name + _hand{idx}
- Save 21 landmark coordinates (x,y,z) to a CSV per saved image (same basename)
- Skip files with no detected hands
- CLI: python preprocess_hands.py --src ../data --dst ../data_preprocessed --img-size 224 --margin 0.2
- Progress displayed with tqdm
"""

import argparse
import os
from pathlib import Path
import sys
import csv
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError as e:
    print("Error: mediapipe is not installed. Please run: pip install mediapipe")
    raise

# Default parameters
DEFAULT_IMG_SIZE = 224
DEFAULT_MARGIN = 0.2  # 20% margin around the bounding box
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def detect_hands_and_landmarks(image_np: np.ndarray, hands) -> List[np.ndarray]:
    """Run MediaPipe Hands on an RGB image and return list of (21,3) landmark arrays.
    Each row is (x, y, z) normalized coordinates; x,y in [0,1] relative to width/height.
    """
    results = hands.process(image_np)
    lm_list = []
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            pts = []
            for lm in hand_lms.landmark:
                pts.append([lm.x, lm.y, lm.z])
            lm_list.append(np.array(pts, dtype=np.float32))
    return lm_list


def landmarks_to_bbox(landmarks_norm: np.ndarray, width: int, height: int, margin: float) -> Tuple[int, int, int, int]:
    """Convert normalized landmarks to a pixel-space bounding box with margin and clamp.
    Returns (x1, y1, x2, y2) as ints.
    """
    xs = (landmarks_norm[:, 0] * width)
    ys = (landmarks_norm[:, 1] * height)
    x1 = float(np.min(xs))
    x2 = float(np.max(xs))
    y1 = float(np.min(ys))
    y2 = float(np.max(ys))

    # Expand by margin
    w = x2 - x1
    h = y2 - y1
    x1 -= w * margin
    y1 -= h * margin
    x2 += w * margin
    y2 += h * margin

    # Ensure min size (in case of extremely tight boxes)
    min_side = 10.0
    if (x2 - x1) < min_side:
        cx = (x1 + x2) / 2
        x1 = cx - min_side / 2
        x2 = cx + min_side / 2
    if (y2 - y1) < min_side:
        cy = (y1 + y2) / 2
        y1 = cy - min_side / 2
        y2 = cy + min_side / 2

    # Clamp to image bounds
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))

    # Fix ordering if needed
    if x2 <= x1: x2 = min(width, x1 + 1)
    if y2 <= y1: y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def crop_and_resize(image_np: np.ndarray, box: Tuple[int, int, int, int], out_size: int) -> Image.Image:
    x1, y1, x2, y2 = box
    crop = image_np[y1:y2, x1:x2, :]
    im = Image.fromarray(crop)
    im = im.resize((out_size, out_size), Image.BILINEAR)
    return im


def save_landmarks_csv(csv_path: Path, landmarks_norm: np.ndarray):
    """Save 21x3 landmark array as CSV with header x,y,z (21 rows)."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])  # header
        for row in landmarks_norm:
            writer.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.6f}"])


def process_dataset(src_dir: Path, dst_dir: Path, img_size: int, margin: float) -> None:
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        sys.exit(1)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Collect class folders
    class_folders = sorted([p for p in src_dir.iterdir() if p.is_dir()])
    total_images = 0
    saved_crops = 0
    skipped_images = 0

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        model_complexity=1,
    ) as hands:
        for class_folder in class_folders:
            class_name = class_folder.name
            dst_class = dst_dir / class_name
            dst_class.mkdir(parents=True, exist_ok=True)

            image_files = []
            for ext in VALID_EXTS:
                image_files.extend(class_folder.glob(f"*{ext}"))
                image_files.extend(class_folder.glob(f"*{ext.upper()}"))

            for img_path in tqdm(image_files, desc=f"{class_name}", unit="img"):
                total_images += 1
                try:
                    im = Image.open(img_path).convert("RGB")
                    w, h = im.size
                    image_np = np.array(im)

                    # MediaPipe expects RGB
                    landmarks_list = detect_hands_and_landmarks(image_np, hands)
                    if not landmarks_list:
                        skipped_images += 1
                        continue

                    # For each detected hand, save crop and CSV
                    base = img_path.stem
                    ext = img_path.suffix.lower()

                    for idx, lms in enumerate(landmarks_list, start=1):
                        box = landmarks_to_bbox(lms, w, h, margin)
                        crop_im = crop_and_resize(image_np, box, img_size)

                        out_name = f"{base}_hand{idx}{ext}"
                        out_img_path = dst_class / out_name
                        crop_im.save(out_img_path)

                        out_csv_path = dst_class / f"{base}_hand{idx}.csv"
                        save_landmarks_csv(out_csv_path, lms)
                        saved_crops += 1
                except Exception as e:
                    # Corrupted or unreadable files -> skip
                    skipped_images += 1
                    continue

    print("\nPreprocessing completed.")
    print(f"Total source images scanned: {total_images}")
    print(f"Total hand crops saved:      {saved_crops}")
    print(f"Total images skipped:        {skipped_images}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset with MediaPipe Hands")
    parser.add_argument("--src", type=str, default="../data", help="Source dataset directory")
    parser.add_argument("--dst", type=str, default="../data_preprocessed", help="Destination directory for preprocessed data")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Output image size (square)")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="Relative margin added around hand box")
    return parser.parse_args()


def main():
    args = parse_args()
    # Resolve relative paths from script location
    script_dir = Path(__file__).resolve().parent
    src_dir = (script_dir / args.src).resolve() if not os.path.isabs(args.src) else Path(args.src)
    dst_dir = (script_dir / args.dst).resolve() if not os.path.isabs(args.dst) else Path(args.dst)

    print(f"Source:      {src_dir}")
    print(f"Destination: {dst_dir}")
    print(f"Img size:    {args.img_size}")
    print(f"Margin:      {args.margin}")

    process_dataset(src_dir, dst_dir, args.img_size, args.margin)


if __name__ == "__main__":
    main()
