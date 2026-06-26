"""
Production-audit evaluation harness (read-only; does not modify models).

Goals:
- Verify whether backend/model_v2.onnx matches backend/best_model.h5 (deployment integrity).
- Quantify the train/inference preprocessing mismatch:
    * full-frame + bilinear  (training-faithful)
    * full-frame + LANCZOS   (REST inference without MediaPipe crop)
    * hand-crop  + LANCZOS   (current default backend inference)
- Produce accuracy / top-3 / mean top-1 confidence / per-class accuracy / confusion matrix.

Usage:
  .venv\\Scripts\\python.exe ML/audit_eval.py --per-class 8 --out ML/audit_out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def load_labels(path: Path):
    with path.open(encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def sample_files(data_dir: Path, class_names, per_class: int):
    files, labels = [], []
    for idx, c in enumerate(class_names):
        cdir = data_dir / c
        if not cdir.is_dir():
            continue
        imgs = sorted([p for p in cdir.iterdir() if p.suffix.lower() in VALID_EXT])
        # Take an evenly spaced sample so we don't just grab the first N near-duplicates.
        if per_class and len(imgs) > per_class:
            step = len(imgs) / per_class
            imgs = [imgs[int(i * step)] for i in range(per_class)]
        for p in imgs:
            files.append(p)
            labels.append(idx)
    return files, np.array(labels, dtype=np.int64)


def normalize(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 127.5 - 1.0


def prep_full(path: Path, size, resample) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size[1], size[0]), resample)
    return normalize(np.asarray(img))


def build_crop_fn():
    """Returns (fn(path)->np.ndarray or None, ok) using backend MediaPipe + bbox logic."""
    try:
        from ensemble_inference import hand_bbox_from_landmarks
        from mediapipe_tasks_hands import (
            TasksHandsCompat,
            ensure_hand_landmarker_model,
            resolve_hand_landmarker_model_path,
        )
    except Exception as e:  # pragma: no cover
        print(f"[crop] unavailable: {e}")
        return None, False

    mp_path = resolve_hand_landmarker_model_path()
    if not mp_path.is_file():
        ensure_hand_landmarker_model(dest=mp_path)
    hands = TasksHandsCompat(
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
        model_path=mp_path,
    )
    state = {"no_hand": 0}

    def fn(path: Path, size, resample):
        img = Image.open(path).convert("RGB")
        arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        H, W, _ = arr.shape
        res = hands.process(arr)
        if not res.multi_hand_landmarks:
            state["no_hand"] += 1
            return None
        lms = list(res.multi_hand_landmarks[0].landmark)
        box = hand_bbox_from_landmarks(lms, W, H, 0.2)
        if box is None:
            state["no_hand"] += 1
            return None
        x1, y1, x2, y2 = box
        crop = arr[y1:y2, x1:x2]
        if crop.size == 0:
            state["no_hand"] += 1
            return None
        crop_img = Image.fromarray(crop).resize((size[1], size[0]), resample)
        return normalize(np.asarray(crop_img))

    return fn, True, state


def run_onnx(session, input_name, X):
    out = session.run(None, {input_name: X})[0]
    return out


def metrics(probs, y_true, class_names):
    y_pred = np.argmax(probs, axis=1)
    acc = float((y_pred == y_true).mean())
    top3 = float(np.mean([y_true[i] in np.argsort(probs[i])[-3:] for i in range(len(y_true))]))
    mean_conf = float(np.mean(probs[np.arange(len(y_true)), y_pred]))
    mean_true_conf = float(np.mean(probs[np.arange(len(y_true)), y_true]))
    # per-class accuracy
    per_class = {}
    for idx, c in enumerate(class_names):
        mask = y_true == idx
        if mask.sum() == 0:
            continue
        per_class[c] = float((y_pred[mask] == idx).mean())
    return {
        "accuracy": acc,
        "top3": top3,
        "mean_top1_confidence": mean_conf,
        "mean_true_class_confidence": mean_true_conf,
        "per_class_accuracy": per_class,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(ROOT / "ML" / "audit_out"))
    ap.add_argument("--data", type=str, default=str(ROOT / "data"))
    ap.add_argument("--skip-h5", action="store_true")
    ap.add_argument("--skip-crop", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)
    labels = load_labels(BACKEND / "class_labels.txt")
    files, y_true = sample_files(data_dir, labels, args.per_class)
    print(f"Sampled {len(files)} images across {len(labels)} classes "
          f"(~{args.per_class}/class)")

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(BACKEND / "model_v2.onnx"), so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    size = (224, 224)

    report = {"n_samples": len(files), "per_class": args.per_class, "modes": {}}

    # ---- full-frame bilinear (training-faithful) ----
    X = np.stack([prep_full(p, size, Image.Resampling.BILINEAR) for p in files])
    probs = run_onnx(sess, in_name, X)
    report["modes"]["onnx_full_bilinear"] = metrics(probs, y_true, labels)
    print("onnx_full_bilinear:", {k: report["modes"]["onnx_full_bilinear"][k]
          for k in ("accuracy", "top3", "mean_top1_confidence")})
    np.save(out_dir / "probs_full_bilinear.npy", probs)

    # ---- full-frame LANCZOS (REST w/o crop) ----
    Xl = np.stack([prep_full(p, size, Image.Resampling.LANCZOS) for p in files])
    probs_l = run_onnx(sess, in_name, Xl)
    report["modes"]["onnx_full_lanczos"] = metrics(probs_l, y_true, labels)
    print("onnx_full_lanczos:", {k: report["modes"]["onnx_full_lanczos"][k]
          for k in ("accuracy", "top3", "mean_top1_confidence")})

    # ---- hand-crop LANCZOS (current default backend inference) ----
    if not args.skip_crop:
        crop = build_crop_fn()
        if crop[1]:
            crop_fn, _, state = crop
            Xc, yc = [], []
            for p, yt in zip(files, y_true):
                a = crop_fn(p, size, Image.Resampling.LANCZOS)
                if a is not None:
                    Xc.append(a)
                    yc.append(yt)
            if Xc:
                Xc = np.stack(Xc)
                yc = np.array(yc)
                probs_c = run_onnx(sess, in_name, Xc)
                m = metrics(probs_c, yc, labels)
                m["n_with_hand"] = int(len(yc))
                m["n_no_hand_dropped"] = int(state["no_hand"])
                report["modes"]["onnx_handcrop_lanczos"] = m
                print("onnx_handcrop_lanczos:", {k: m[k] for k in
                      ("accuracy", "top3", "mean_top1_confidence")},
                      f"(hand found {len(yc)}/{len(files)})")

    # ---- h5 vs onnx identity check (full-frame bilinear) ----
    if not args.skip_h5:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(BACKEND / "best_model.h5"), compile=False)
            n = min(len(files), 64)
            h5_probs = model.predict(X[:n], verbose=0)
            h5_pred = np.argmax(h5_probs, axis=1)
            onnx_pred = np.argmax(probs[:n], axis=1)
            agree = float((h5_pred == onnx_pred).mean())
            h5_m = metrics(h5_probs, y_true[:n], labels)
            report["h5_vs_onnx"] = {
                "compared_on": n,
                "prediction_agreement": agree,
                "h5_accuracy_full_bilinear": h5_m["accuracy"],
                "h5_top3": h5_m["top3"],
                "h5_mean_top1_confidence": h5_m["mean_top1_confidence"],
                "h5_param_count": int(model.count_params()),
            }
            print(f"h5 params={model.count_params():,}  "
                  f"h5_vs_onnx agreement={agree:.3f}  "
                  f"h5_acc={h5_m['accuracy']:.3f}")
        except Exception as e:
            report["h5_vs_onnx"] = {"error": str(e)}
            print(f"[h5] check failed: {e}")

    with (out_dir / "audit_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_dir / 'audit_report.json'}")


if __name__ == "__main__":
    main()
