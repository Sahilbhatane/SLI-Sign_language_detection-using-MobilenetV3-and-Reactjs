"""
MediaPipe HandLandmarker (Tasks API) with a small compatibility layer for code
that expected the removed `mediapipe.solutions.hands` API (landmark objects
with .x / .y / .z and results.multi_hand_landmarks).

Model: Google-hosted hand_landmarker.task (downloaded once under backend/.cache/
unless MEDIAPIPE_HAND_MODEL_PATH points to an existing file).
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Official bundle used by MediaPipe Tasks Hand Landmarker (float16 / v1).
DEFAULT_HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _backend_dir() -> Path:
    return Path(__file__).resolve().parent


def default_task_cache_path() -> Path:
    return _backend_dir() / ".cache" / "hand_landmarker.task"


def resolve_hand_landmarker_model_path() -> Path:
    """
    Return path to hand_landmarker.task.
    Order: MEDIAPIPE_HAND_MODEL_PATH if set and exists; else cached default path.
    Does not download; caller should call ensure_hand_landmarker_model().
    """
    env = os.environ.get("MEDIAPIPE_HAND_MODEL_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()
        logger.warning(
            "MEDIAPIPE_HAND_MODEL_PATH is set but file not found: %s — using cache path",
            p,
        )
    return default_task_cache_path()


def ensure_hand_landmarker_model(
    url: str = DEFAULT_HAND_TASK_URL,
    dest: Optional[Path] = None,
    timeout_s: int = 120,
) -> Path:
    """
    Ensure hand_landmarker.task exists at dest (default: backend/.cache/).
    Downloads from url if missing. Returns resolved path.
    """
    path = dest or default_task_cache_path()
    path = path.resolve()
    if path.is_file():
        logger.info("Hand landmarker model present at %s (%s bytes)", path, path.stat().st_size)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    logger.info("Downloading hand landmarker model from %s -> %s", url, path)
    req = urllib.request.Request(url, headers={"User-Agent": "SLI-sign-language/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
        partial.write_bytes(data)
        partial.replace(path)
    except Exception:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
        raise
    logger.info("Downloaded hand landmarker model (%s bytes) to %s", path.stat().st_size, path)
    return path


@dataclass
class _Lm:
    x: float
    y: float
    z: float


@dataclass
class _HandLm:
    landmark: List[_Lm]


def tasks_result_to_multi_hand_landmarks(result) -> Optional[List[_HandLm]]:
    """Convert HandLandmarkerResult to list compatible with legacy solutions.hands."""
    if not result.hand_landmarks:
        return None
    out: List[_HandLm] = []
    for hand_lms in result.hand_landmarks:
        pts = []
        for lm in hand_lms:
            pts.append(
                _Lm(
                    float(lm.x) if lm.x is not None else 0.0,
                    float(lm.y) if lm.y is not None else 0.0,
                    float(lm.z) if lm.z is not None else 0.0,
                )
            )
        out.append(_HandLm(landmark=pts))
    return out


class TasksHandsCompat:
    """
    Drop-in subset of mp.solutions.hands.Hands: .process(rgb_uint8) -> object with
    .multi_hand_landmarks; supports context manager for cleanup.
    """

    def __init__(
        self,
        *,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.25,
        min_hand_presence_confidence: float = 0.25,
        min_tracking_confidence: float = 0.25,
        model_path: Optional[Path] = None,
    ) -> None:
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.vision.core import vision_task_running_mode

        mp_path = model_path or ensure_hand_landmarker_model()
        opts = HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(mp_path)),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            num_hands=int(num_hands),
            min_hand_detection_confidence=float(min_hand_detection_confidence),
            min_hand_presence_confidence=float(min_hand_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._landmarker = HandLandmarker.create_from_options(opts)
        from mediapipe.tasks.python.vision.core import image as mp_image_module

        self._mp_image_module = mp_image_module

    def process(self, image_np: np.ndarray) -> SimpleNamespace:
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        if not image_np.flags["C_CONTIGUOUS"]:
            image_np = np.ascontiguousarray(image_np)
        fmt = self._mp_image_module.ImageFormat.SRGB
        mp_image = self._mp_image_module.Image(fmt, image_np)
        result = self._landmarker.detect(mp_image)
        lms = tasks_result_to_multi_hand_landmarks(result)
        return SimpleNamespace(multi_hand_landmarks=lms)

    def close(self) -> None:
        lm = getattr(self, "_landmarker", None)
        if lm is not None:
            lm.close()
            self._landmarker = None

    def __enter__(self) -> "TasksHandsCompat":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def log_mediapipe_runtime_diagnostics() -> None:
    """One-line evidence for support tickets (version + solutions attr)."""
    try:
        import mediapipe as mp

        ver = getattr(mp, "__version__", "unknown")
        has_solutions = hasattr(mp, "solutions")
        logger.info(
            "MediaPipe package: version=%s top_level_has_solutions=%s (0.10.30+ uses Tasks API only)",
            ver,
            has_solutions,
        )
    except Exception as e:
        logger.warning("Could not introspect mediapipe package: %s", e)
