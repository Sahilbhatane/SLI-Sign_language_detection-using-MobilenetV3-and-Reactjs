"""
Verify the REAL backend inference pipeline (ensemble_inference.ONNXSignLanguageModel)
end-to-end after the crop/preprocessing fixes.

Scores a balanced dataset sample through model.preprocess_image + model.predict,
i.e. the exact code path the FastAPI /predict and WebRTC endpoints use.

Run with hand gating off so every sampled frame is scored apples-to-apples:
  ENABLE_HAND_GATING=0  .venv\\Scripts\\python.exe ML/verify_backend_pipeline.py --per-class 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6)
    args = ap.parse_args()

    from ensemble_inference import ONNXSignLanguageModel

    model = ONNXSignLanguageModel()
    labels = model.class_labels
    crop = os.environ.get("ENABLE_MEDIAPIPE_CROP", "0")
    gating = os.environ.get("ENABLE_HAND_GATING", "1")
    print(f"crop={crop} gating={gating} smoothing_window={model.pred_history.maxlen}")

    data_dir = ROOT / "data"
    y_true, y_pred, confs = [], [], []
    n_gated = 0
    for idx, c in enumerate(labels):
        cdir = data_dir / c
        if not cdir.is_dir():
            continue
        imgs = sorted([p for p in cdir.iterdir() if p.suffix.lower() in VALID_EXT])
        if args.per_class and len(imgs) > args.per_class:
            step = len(imgs) / args.per_class
            imgs = [imgs[int(i * step)] for i in range(args.per_class)]
        for p in imgs:
            arr = model.preprocess_image(Image.open(p))
            if not getattr(model, "last_hand_detected", True):
                n_gated += 1
            probs = model.predict(arr)
            pi = int(np.argmax(probs))
            y_true.append(idx)
            y_pred.append(pi)
            confs.append(float(probs[pi]))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    print(f"samples={len(y_true)}  accuracy={acc:.4f}  "
          f"mean_top1_conf={np.mean(confs):.4f}  gated(no-hand)={n_gated}")


if __name__ == "__main__":
    main()
