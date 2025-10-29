"""
Temperature scaling calibration utilities.

Provides a simple function to estimate the optimal temperature (T) by minimizing
negative log-likelihood on validation probabilities and true labels.
Optionally runnable as a script if you have probabilities and labels saved in an NPZ.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar


def _softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def apply_temperature_scaling(arr: np.ndarray, temperature: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    was_1d = False
    if arr.ndim == 1:
        arr = arr[None, :]
        was_1d = True

    # Detect if probabilities
    row_sums = np.sum(arr, axis=1)
    is_prob = np.all(np.isfinite(row_sums)) and np.allclose(row_sums, 1.0, atol=1e-3)
    if is_prob:
        eps = 1e-12
        logit_like = np.log(np.clip(arr, eps, 1.0))
    else:
        logit_like = arr

    scaled = logit_like / float(max(temperature, 1e-6))
    probs = _softmax_np(scaled, axis=1)
    return probs[0] if was_1d else probs


def calibrate_temperature(probs: np.ndarray, y_true: np.ndarray, bounds: Tuple[float, float] = (0.1, 5.0)) -> Tuple[float, float, float]:
    """
    Calibrate temperature using validation set probabilities and labels.

    Args:
        probs: (N, C) predicted probabilities or logits
        y_true: (N,) true integer labels
        bounds: (low, high) search bounds for T

    Returns:
        best_T, old_avg_conf, new_avg_conf
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true, dtype=np.int64)

    def nll_for_T(T: float) -> float:
        cal = apply_temperature_scaling(probs, max(T, 1e-6))
        eps = 1e-12
        p_true = cal[np.arange(cal.shape[0]), y_true]
        return float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))

    res = minimize_scalar(nll_for_T, bounds=bounds, method='bounded')
    best_T = float(res.x)

    old_conf = float(np.mean(np.max(probs if np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-3) else _softmax_np(probs), axis=1)))
    new_probs = apply_temperature_scaling(probs, best_T)
    new_conf = float(np.mean(np.max(new_probs, axis=1)))
    return best_T, old_conf, new_conf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate temperature from saved probs and labels (NPZ with 'probs' and 'labels').")
    parser.add_argument("npz_path", help="Path to npz file containing 'probs' and 'labels'")
    parser.add_argument("--save-json", default="backend/temp_scale.json", help="Where to save temperature json")
    args = parser.parse_args()

    data = np.load(args.npz_path)
    probs = data["probs"]
    labels = data["labels"]

    T, old_c, new_c = calibrate_temperature(probs, labels)
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump({"temperature": round(T, 6)}, f)
    print(f"Best T: {T:.4f}\nAvg top-1 confidence (old): {old_c:.4f}\nAvg top-1 confidence (new): {new_c:.4f}\nSaved: {args.save_json}")
