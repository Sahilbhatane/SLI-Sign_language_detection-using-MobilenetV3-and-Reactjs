"""
Benchmark MediaPipe HandLandmarker latency: IMAGE vs VIDEO running mode, and the
effect of downscaling the detection input. Landmarks are normalized (0-1), so
detecting on a downscaled frame does not change overlay/crop coordinates.

Uses real sequential frames from data/<class> so VIDEO-mode tracking is exercised.

  .venv\\Scripts\\python.exe ML/bench_mediapipe.py --frames 40 --class stop
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def load_frames(class_name: str, n: int):
    cdir = ROOT / "data" / class_name
    imgs = sorted([p for p in cdir.iterdir() if p.suffix.lower() in VALID_EXT])[:n]
    return [np.ascontiguousarray(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
            for p in imgs]


def downscale(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    m = max(h, w)
    if max_side <= 0 or m <= max_side:
        return arr
    scale = max_side / float(m)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = Image.fromarray(arr).resize((nw, nh), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def make_landmarker(mode: str, model_path, num_hands=2):
    from mediapipe.tasks.python.core import base_options as bo
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as vm

    mode_map = {
        "image": vm.VisionTaskRunningMode.IMAGE,
        "video": vm.VisionTaskRunningMode.VIDEO,
    }
    opts = HandLandmarkerOptions(
        base_options=bo.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mode_map[mode],
        num_hands=num_hands,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    return HandLandmarker.create_from_options(opts)


def bench(mode: str, frames, max_side: int, model_path):
    from mediapipe.tasks.python.vision.core import image as mp_image_module
    fmt = mp_image_module.ImageFormat.SRGB
    lm = make_landmarker(mode, model_path)

    prepped = [downscale(f, max_side) for f in frames]
    # warmup
    for i, f in enumerate(prepped[:3]):
        img = mp_image_module.Image(fmt, f)
        if mode == "video":
            lm.detect_for_video(img, i)
        else:
            lm.detect(img)

    n_hands = 0
    t0 = time.perf_counter()
    ts = 1000
    for f in prepped:
        img = mp_image_module.Image(fmt, f)
        if mode == "video":
            ts += 33
            res = lm.detect_for_video(img, ts)
        else:
            res = lm.detect(img)
        if res.hand_landmarks:
            n_hands += 1
    dt = (time.perf_counter() - t0) / len(prepped) * 1000
    lm.close()
    return dt, n_hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--class", dest="cls", type=str, default="stop")
    args = ap.parse_args()

    from mediapipe_tasks_hands import (
        ensure_hand_landmarker_model,
        resolve_hand_landmarker_model_path,
    )
    mp_path = resolve_hand_landmarker_model_path()
    if not mp_path.is_file():
        ensure_hand_landmarker_model(dest=mp_path)

    frames = load_frames(args.cls, args.frames)
    print(f"Loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]}) from '{args.cls}'\n")

    configs = [
        ("IMAGE  full(640)", "image", 0),
        ("IMAGE  down(320)", "image", 320),
        ("IMAGE  down(256)", "image", 256),
        ("VIDEO  full(640)", "video", 0),
        ("VIDEO  down(320)", "video", 320),
        ("VIDEO  down(256)", "video", 256),
    ]
    print(f"{'config':18}  {'ms/frame':>9}  {'FPS':>6}  {'hand_found':>10}")
    print("-" * 50)
    for name, mode, ms in configs:
        dt, nh = bench(mode, frames, ms, mp_path)
        print(f"{name:18}  {dt:9.1f}  {1000/dt:6.1f}  {nh:>4}/{len(frames)}")


if __name__ == "__main__":
    main()
