"""
Ensemble ONNX inference utilities for Sign Language Recognition.
Contains ONNXSignLanguageModel used by the FastAPI app.
"""

import os
import json
import base64
import logging
from collections import deque
from io import BytesIO
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from PIL import Image
import onnxruntime as ort

logger = logging.getLogger(__name__)


def _downscale_for_detection(image_np: np.ndarray, max_side: int) -> np.ndarray:
    """
    Downscale an RGB frame so its longest side is <= max_side, for MediaPipe detection.
    MediaPipe landmarks are normalized (0-1), so overlay/crop coordinates are unaffected.
    Detecting near the palm model's trained resolution is faster AND more reliable than
    full 640px (benchmarked in ML/bench_mediapipe.py). max_side <= 0 disables downscaling.
    """
    if max_side <= 0:
        return image_np
    h, w = image_np.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_np
    scale = max_side / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = Image.fromarray(image_np).resize((nw, nh), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))

# ~20% padding on hand axis-aligned bounding box (from 21 keypoints)
HAND_BBOX_PAD_FRAC = 0.2


def hand_bbox_union(
    boxes: List[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """Axis-aligned union of hand boxes (for two-handed signs / overlay)."""
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def hand_bbox_from_landmarks(
    landmark_list, width: int, height: int, pad_frac: float = HAND_BBOX_PAD_FRAC
) -> Optional[Tuple[int, int, int, int]]:
    """
    Build a tight axis-aligned box around MediaPipe hand landmarks and expand by pad_frac.
    Returns (x1, y1, x2, y2) in pixel coords with y2/x2 exclusive upper bounds, or None.
    """
    if not landmark_list or width <= 0 or height <= 0:
        return None
    xs = [float(lm.x) * width for lm in landmark_list]
    ys = [float(lm.y) * height for lm in landmark_list]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 and bh <= 0:
        return None
    px = (bw * pad_frac) if bw > 0 else (pad_frac * 10.0)
    py = (bh * pad_frac) if bh > 0 else (pad_frac * 10.0)
    x1e = int(np.floor(x1 - px))
    y1e = int(np.floor(y1 - py))
    x2e = int(np.ceil(x2 + px))
    y2e = int(np.ceil(y2 + py))
    x1c = int(np.clip(x1e, 0, width - 1))
    y1c = int(np.clip(y1e, 0, height - 1))
    x2c = int(np.clip(x2e, x1c + 1, width))
    y2c = int(np.clip(y2e, y1c + 1, height))
    return (x1c, y1c, x2c, y2c)


def wrist_relative_hand_landmarks(landmark_list) -> np.ndarray:
    """
    21x3 array: each row is (x, y, z) relative to wrist (index 0), in MediaPipe normalized space.
    """
    out = np.zeros((21, 3), dtype=np.float32)
    if not landmark_list or len(landmark_list) < 21:
        return out
    wrist = landmark_list[0]
    for i, lm in enumerate(landmark_list):
        if i >= 21:
            break
        out[i, 0] = float(lm.x) - float(wrist.x)
        out[i, 1] = float(lm.y) - float(wrist.y)
        out[i, 2] = float(lm.z) - float(wrist.z)
    return out


def overlay_hands_from_landmark_results(
    multi_hand_landmarks, width: int, height: int, pad_frac: float = HAND_BBOX_PAD_FRAC
) -> List[Dict[str, Any]]:
    """
    Build per-hand overlay dicts: bbox_norm + landmarks_norm (0–1 image space).
    """
    if not multi_hand_landmarks or width <= 0 or height <= 0:
        return []
    hands_out: List[Dict[str, Any]] = []
    wf, hf = float(width), float(height)
    for hand_lms in multi_hand_landmarks:
        lms = list(hand_lms.landmark)
        box = hand_bbox_from_landmarks(lms, width, height, pad_frac)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        hands_out.append({
            "bbox_norm": [x1 / wf, y1 / hf, x2 / wf, y2 / hf],
            "landmarks_norm": [[float(lm.x), float(lm.y)] for lm in lms[:21]],
        })
    return hands_out


def mirror_overlay_hands_norm_x(hands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flip normalized x for selfie-mirrored retry (map back to original camera frame)."""
    mirrored: List[Dict[str, Any]] = []
    for h in hands:
        b = h.get("bbox_norm")
        lms = h.get("landmarks_norm")
        entry: Dict[str, Any] = {}
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            entry["bbox_norm"] = [1.0 - x2, y1, 1.0 - x1, y2]
        if isinstance(lms, list) and lms:
            entry["landmarks_norm"] = [[1.0 - float(p[0]), float(p[1])] for p in lms[:21]]
        if entry:
            mirrored.append(entry)
    return mirrored


class ONNXSignLanguageModel:
    """
    ONNX Sign Language Recognition Model Handler
    Supports optional MediaPipe-based cropping and an optional landmark model.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None, labels_path: str = None, enable_mediapipe_crop: bool = True, smoothing_window: int = 1):
        # Only initialize once
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Set default paths (env overrides; prefer model_v2.onnx with legacy fallback)
        if model_path is None:
            env_model_path = os.getenv("MODEL_PATH", "").strip()
            if env_model_path:
                model_path = env_model_path
            else:
                default_model_path = os.path.join(os.path.dirname(__file__), "model_v2.onnx")
                legacy_model_path = os.path.join(os.path.dirname(__file__), "model.onnx")
                if os.path.exists(default_model_path):
                    model_path = default_model_path
                elif os.path.exists(legacy_model_path):
                    model_path = legacy_model_path
                else:
                    model_path = default_model_path
        if labels_path is None:
            env_labels_path = os.getenv("LABELS_PATH", "").strip()
            labels_path = env_labels_path or os.path.join(os.path.dirname(__file__), "class_labels.txt")

        self.model_path = model_path
        self.labels_path = labels_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        logger.info(f"Loading ONNX model from: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CPU latency tuning (onnxruntime.ai threading guide). All env-overridable.
        try:
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            _intra = os.environ.get("ORT_INTRA_OP_THREADS")
            if _intra is not None:
                sess_options.intra_op_num_threads = max(0, int(_intra))
            sess_options.add_session_config_entry("session.dynamic_block_base", "4")
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        except Exception as e:
            logger.warning("ORT session tuning skipped: %s", e)

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            logger.info(f"Model loaded with provider: {self.session.get_providers()[0]}")
        except Exception as e:
            logger.warning(f"Failed to load with GPU, using CPU: {e}")
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.target_size = self._infer_target_size(self.input_shape)

        self.sessions = [
            {
                'type': 'image',
                'session': self.session,
                'input_name': self.input_name,
                'input_shape': self.input_shape,
            }
        ]

        # Optional second model for landmarks
        self.landmark_session = None
        self.landmark_input_name = None
        self.landmark_input_shape = None
        landmark_path = os.path.join(os.path.dirname(__file__), "landmark_model.onnx")
        if os.path.exists(landmark_path):
            try:
                lm_sess = ort.InferenceSession(
                    landmark_path,
                    sess_options=sess_options,
                    providers=providers
                )
                self.landmark_session = lm_sess
                self.landmark_input_name = lm_sess.get_inputs()[0].name
                self.landmark_input_shape = lm_sess.get_inputs()[0].shape
                self.sessions.append({
                    'type': 'landmark',
                    'session': lm_sess,
                    'input_name': self.landmark_input_name,
                    'input_shape': self.landmark_input_shape,
                })
                logger.info(f"Loaded optional landmark model: {landmark_path}")
            except Exception as e:
                logger.warning(f"Failed to load landmark model: {e}")

        with open(labels_path, 'r', encoding='utf-8') as f:
            self.class_labels = [line.strip() for line in f.readlines()]
        self.num_classes = len(self.class_labels)

        # IMPORTANT (accuracy): the classifier is trained on FULL upper-body frames
        # (see ML/train_efficientnetv2.py -> image_dataset_from_directory on data/).
        # Feeding it a tight MediaPipe hand crop is out-of-distribution and collapses
        # accuracy (measured 98.5% full-frame vs 27.1% hand-crop on the dataset).
        # Crop is therefore OFF by default; MediaPipe is still used for the live
        # overlay and hand-presence gating. Only enable crop with a model that was
        # explicitly trained on hand crops (ML/preprocess_hands.py -> data_preprocessed/).
        _env = os.environ.get("ENABLE_MEDIAPIPE_CROP", "0").strip()
        self.enable_mediapipe_crop = bool(enable_mediapipe_crop) and _env == "1"
        # When crop is off we still gate predictions on hand presence so empty frames
        # don't yield confident garbage on a live stream. Set ENABLE_HAND_GATING=0 to
        # always predict on the full frame (e.g. for offline dataset scoring).
        self.hand_gating = os.environ.get("ENABLE_HAND_GATING", "1").strip() != "0"
        self._mp_hands = None
        self.last_hand_detected: bool = True
        self.last_wrist_norm_landmarks: Optional[np.ndarray] = None
        self._last_mp_raw_landmark_vec: Optional[np.ndarray] = None
        # Normalized 0–1 in full input image (for client overlay on same frame as inference)
        self.last_overlay_bbox_norm: Optional[Tuple[float, float, float, float]] = None
        self.last_overlay_landmarks_norm: Optional[List[List[float]]] = None
        self.last_overlay_hands_norm: List[Dict[str, Any]] = []
        _num_hands_env = os.environ.get("MEDIAPIPE_NUM_HANDS", "2").strip()
        try:
            self.mediapipe_num_hands = max(1, min(2, int(_num_hands_env)))
        except ValueError:
            self.mediapipe_num_hands = 2
        # VIDEO running mode (temporal tracking) is ~1.6-1.9x faster on CPU for a live
        # stream than IMAGE mode. Default to video; set MEDIAPIPE_RUNNING_MODE=image for
        # scoring unrelated still frames. Downscaling detection input to ~320px is faster
        # and more reliable than full 640px (landmarks are normalized, so overlay/crop
        # coordinates are unchanged). MEDIAPIPE_MAX_SIDE=0 disables downscaling.
        self.mediapipe_running_mode = os.environ.get("MEDIAPIPE_RUNNING_MODE", "video").strip().lower()
        if self.mediapipe_running_mode not in ("image", "video"):
            self.mediapipe_running_mode = "video"
        _max_side_env = os.environ.get("MEDIAPIPE_MAX_SIDE", "320").strip()
        try:
            self.mediapipe_max_side = max(0, int(_max_side_env))
        except ValueError:
            self.mediapipe_max_side = 320
        # Always try to load Hands so WebRTC overlay works even when ENABLE_MEDIAPIPE_CROP=0.
        # mediapipe>=0.10.30 wheels removed `mediapipe.solutions`; use Tasks HandLandmarker.
        try:
            from mediapipe_tasks_hands import (
                TasksHandsCompat,
                ensure_hand_landmarker_model,
                log_mediapipe_runtime_diagnostics,
                resolve_hand_landmarker_model_path,
            )

            log_mediapipe_runtime_diagnostics()
            model_path = resolve_hand_landmarker_model_path()
            if not model_path.is_file():
                ensure_hand_landmarker_model(dest=model_path)
            logger.info(
                "Initializing MediaPipe Tasks HandLandmarker (model=%s, mode=%s, max_side=%d)",
                model_path, self.mediapipe_running_mode, self.mediapipe_max_side,
            )
            self._mp_hands = TasksHandsCompat(
                num_hands=self.mediapipe_num_hands,
                min_hand_detection_confidence=0.25,
                min_hand_presence_confidence=0.25,
                min_tracking_confidence=0.25,
                model_path=model_path,
                running_mode=self.mediapipe_running_mode,
            )
            if self.enable_mediapipe_crop:
                logger.info(
                    "MediaPipe hand detection + crop enabled (IMAGE mode, num_hands=%d)",
                    self.mediapipe_num_hands,
                )
            elif self.landmark_session is not None:
                logger.info("MediaPipe landmark vectors enabled (landmark model loaded)")
            else:
                logger.info(
                    "MediaPipe Hands loaded for live overlay (crop disabled, num_hands=%d)",
                    self.mediapipe_num_hands,
                )
        except Exception as e:
            logger.warning("Failed to initialize MediaPipe: %s", e, exc_info=True)
            self._mp_hands = None
            self.enable_mediapipe_crop = False

        self.temperature = 1.0
        self._load_temperature()

        # Backend inference is stateless by default (window=1). A single global model
        # instance serves REST and WebRTC for all clients, so a shared rolling buffer
        # would average probabilities across unrelated frames/sessions and corrupt
        # results. Temporal stabilization is done per-stream on the frontend. Set
        # PRED_SMOOTHING_WINDOW > 1 only for a single-client live setup.
        _sw_env = os.environ.get("PRED_SMOOTHING_WINDOW")
        if _sw_env is not None:
            try:
                smoothing_window = int(_sw_env)
            except ValueError:
                logger.warning("Invalid PRED_SMOOTHING_WINDOW=%r; using %d", _sw_env, smoothing_window)
        self.pred_history = deque(maxlen=max(1, int(smoothing_window)))

        logger.info("✓ Model loaded successfully")
        logger.info(f"  - Input shape: {self.input_shape}")
        logger.info(f"  - Target size for preprocessing: {self.target_size}")
        logger.info(f"  - Classes: {self.num_classes}")
        logger.info(f"  - Provider: {self.session.get_providers()[0]}")

        self._initialized = True

    def _sync_legacy_overlay_fields(self) -> None:
        """Keep single-hand legacy fields in sync with last_overlay_hands_norm."""
        hands = self.last_overlay_hands_norm or []
        if not hands:
            self.last_overlay_bbox_norm = None
            self.last_overlay_landmarks_norm = None
            return
        xs1 = [float(h["bbox_norm"][0]) for h in hands if h.get("bbox_norm")]
        ys1 = [float(h["bbox_norm"][1]) for h in hands if h.get("bbox_norm")]
        xs2 = [float(h["bbox_norm"][2]) for h in hands if h.get("bbox_norm")]
        ys2 = [float(h["bbox_norm"][3]) for h in hands if h.get("bbox_norm")]
        if xs1 and ys1 and xs2 and ys2:
            self.last_overlay_bbox_norm = (min(xs1), min(ys1), max(xs2), max(ys2))
        first_lms = hands[0].get("landmarks_norm")
        self.last_overlay_landmarks_norm = (
            [[float(p[0]), float(p[1])] for p in first_lms[:21]]
            if isinstance(first_lms, list) and first_lms
            else None
        )

    def get_overlay_hands_json(self) -> List[Dict[str, Any]]:
        """JSON-serializable per-hand overlay for REST/WebRTC clients."""
        out: List[Dict[str, Any]] = []
        for h in self.last_overlay_hands_norm or []:
            b = h.get("bbox_norm")
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            lms = h.get("landmarks_norm")
            entry: Dict[str, Any] = {
                "bbox_norm": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
            }
            if isinstance(lms, list) and lms:
                entry["landmarks_norm"] = [[float(p[0]), float(p[1])] for p in lms[:21]]
            out.append(entry)
        return out

    def _infer_target_size(self, input_shape: list) -> Tuple[int, int]:
        try:
            h = None
            w = None
            if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3:
                ints = [d for d in input_shape if isinstance(d, int)]
                if len(ints) >= 2:
                    h, w = ints[0], ints[1]
            if not h or not w:
                return (224, 224)
            return (int(h), int(w))
        except Exception:
            return (224, 224)

    def _load_temperature(self):
        candidates = [
            os.path.join(os.path.dirname(__file__), 'temp_scale.json'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp_scale.json')
        ]
        for path in candidates:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        t = float(data.get('temperature', 1.0))
                        if t > 0:
                            self.temperature = t
                            logger.info(f"Temperature scaling enabled (T={self.temperature}) from {path}")
                            return
            except Exception as e:
                logger.warning(f"Failed to read temperature from {path}: {e}")

    def _extract_landmarks(self, image: Image.Image) -> Optional[np.ndarray]:
        """Run MediaPipe on a PIL image and return 63-dim (x,y,z)*21 vector (first hand)."""
        if self._mp_hands is None:
            return None
        try:
            img_np = np.array(image, dtype=np.uint8)
            if img_np.ndim != 3 or img_np.shape[2] != 3:
                return None
            results = self._mp_hands.process(img_np)
            if not results.multi_hand_landmarks:
                return None
            hand_lms = results.multi_hand_landmarks[0]
            pts = [v for lm in hand_lms.landmark for v in (lm.x, lm.y, lm.z)]
            return np.array(pts, dtype=np.float32)
        except Exception:
            return None

    def preprocess_image(self, image: Image.Image, target_size: tuple = None) -> np.ndarray:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            self.last_wrist_norm_landmarks = None
            self._last_mp_raw_landmark_vec = None
            self.last_overlay_bbox_norm = None
            self.last_overlay_landmarks_norm = None
            self.last_overlay_hands_norm = []
            ts = target_size or self.target_size or (224, 224)
            h_out, w_out = int(ts[0]), int(ts[1])

            # Run MediaPipe whenever loaded so WebRTC overlay matches detection (even if crop is off).
            image_np = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
            if self._mp_hands is not None and image_np.ndim == 3 and image_np.shape[2] == 3:
                try:
                    H, W, _ = image_np.shape
                    # Detect on a downscaled copy (faster + more reliable); landmarks are
                    # normalized so all bbox/overlay/crop math below uses full W, H.
                    det_np = _downscale_for_detection(image_np, self.mediapipe_max_side)
                    results = self._mp_hands.process(det_np)
                    if results.multi_hand_landmarks:
                        self.last_overlay_hands_norm = overlay_hands_from_landmark_results(
                            results.multi_hand_landmarks, W, H, HAND_BBOX_PAD_FRAC
                        )
                        self._sync_legacy_overlay_fields()

                        # Legacy landmark vectors: primary (first) hand
                        hand_lms = results.multi_hand_landmarks[0]
                        lms = list(hand_lms.landmark)
                        self.last_wrist_norm_landmarks = wrist_relative_hand_landmarks(lms)
                        self._last_mp_raw_landmark_vec = np.array(
                            [v for lm in lms for v in (lm.x, lm.y, lm.z)], dtype=np.float32
                        )

                        pixel_boxes: List[Tuple[int, int, int, int]] = []
                        for hand in results.multi_hand_landmarks:
                            b = hand_bbox_from_landmarks(
                                list(hand.landmark), W, H, HAND_BBOX_PAD_FRAC
                            )
                            if b is not None:
                                pixel_boxes.append(b)
                        union_box = hand_bbox_union(pixel_boxes)

                        if self.enable_mediapipe_crop:
                            if union_box is None:
                                self.last_hand_detected = False
                                self.last_overlay_hands_norm = []
                                self.last_overlay_bbox_norm = None
                                self.last_overlay_landmarks_norm = None
                                img_std = np.full((h_out, w_out, 3), -1.0, dtype=np.float32)
                                return np.expand_dims(img_std, axis=0)
                            x1, y1, x2, y2 = union_box
                            crop = image_np[y1:y2, x1:x2]
                            if crop.size == 0:
                                self.last_hand_detected = False
                                self.last_overlay_hands_norm = []
                                self.last_overlay_bbox_norm = None
                                self.last_overlay_landmarks_norm = None
                                img_std = np.full((h_out, w_out, 3), -1.0, dtype=np.float32)
                                return np.expand_dims(img_std, axis=0)
                            image = Image.fromarray(crop)
                            self.last_hand_detected = True
                        else:
                            self.last_hand_detected = True
                    elif self.enable_mediapipe_crop:
                        self.last_hand_detected = False
                        self.last_overlay_hands_norm = []
                        img_std = np.full((h_out, w_out, 3), -1.0, dtype=np.float32)
                        return np.expand_dims(img_std, axis=0)
                    else:
                        self.last_overlay_hands_norm = []
                        # No hand found, crop disabled: gate to "Detecting..." when
                        # hand gating is on; otherwise predict on the full frame.
                        self.last_hand_detected = not self.hand_gating
                except Exception as e:
                    logger.debug(f"MediaPipe failed: {e}")
                    if self.enable_mediapipe_crop:
                        self.last_hand_detected = False
                        self.last_overlay_hands_norm = []
                        img_std = np.full((h_out, w_out, 3), -1.0, dtype=np.float32)
                        return np.expand_dims(img_std, axis=0)
                    # Transient MediaPipe error: keep predicting on the full frame.
                    self.last_hand_detected = True
            else:
                self.last_hand_detected = True

            # target_size is (H, W); PIL resize expects (W, H).
            # BILINEAR matches training (keras image_dataset_from_directory default),
            # keeping inference preprocessing identical to training preprocessing.
            image = image.resize((w_out, h_out), Image.Resampling.BILINEAR)

            # Match training: image / 127.5 - 1.0 -> [-1, 1]. This is a linear float32 op,
            # so NumPy is bit-equivalent to the TF training path (see ML/test_preprocess_norm.py)
            # while avoiding TensorFlow import/convert overhead on every frame.
            img = np.asarray(image, dtype=np.float32)
            img_std = img / 127.5 - 1.0
            img_std = np.expand_dims(img_std, axis=0)
            return img_std
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def preprocess_base64(self, base64_str: str, target_size: tuple = (224, 224)) -> np.ndarray:
        try:
            if ',' in base64_str:
                base64_str = base64_str.split(',', 1)[1]
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_bytes))
            return self.preprocess_image(image, target_size)
        except Exception as e:
            logger.error(f"Error preprocessing base64 image: {e}")
            raise

    def predict(self, image_array: np.ndarray) -> np.ndarray:
        try:
            if not getattr(self, "last_hand_detected", True):
                u = np.full(self.num_classes, 1.0 / float(self.num_classes), dtype=np.float32)
                return u

            probs_list = []

            outputs = self.session.run(None, {self.input_name: image_array})
            raw_img = outputs[0][0]
            probs_list.append(np.clip(raw_img, 1e-8, 1.0))

            if self.landmark_session is not None:
                lm_vec = self._last_mp_raw_landmark_vec
                if lm_vec is None:
                    try:
                        approx = image_array[0]
                        approx = approx - approx.min()
                        if approx.max() > 0:
                            approx = approx / approx.max()
                        approx = (approx * 255.0).astype(np.uint8)
                        pil = Image.fromarray(approx)
                        lm_vec = self._extract_landmarks(pil)
                    except Exception:
                        lm_vec = None

                if lm_vec is not None:
                    inp_shape = self.landmark_input_shape
                    if isinstance(inp_shape, (list, tuple)) and len(inp_shape) == 2:
                        D = inp_shape[1] if isinstance(inp_shape[1], int) else lm_vec.shape[0]
                        if D == lm_vec.shape[0]:
                            lm_inp = lm_vec.reshape(1, D).astype(np.float32)
                        else:
                            lm_inp = None
                    elif isinstance(inp_shape, (list, tuple)) and len(inp_shape) == 3:
                        if inp_shape[1] == 21 and inp_shape[2] == 3:
                            lm_inp = lm_vec.reshape(1, 21, 3).astype(np.float32)
                        else:
                            lm_inp = None
                    else:
                        lm_inp = lm_vec.reshape(1, -1).astype(np.float32)

                    if lm_inp is not None:
                        try:
                            out_lm = self.landmark_session.run(None, {self.landmark_input_name: lm_inp})
                            probs_list.append(np.clip(out_lm[0][0], 1e-8, 1.0))
                        except Exception as e:
                            logger.debug(f"Landmark model inference failed: {e}")

            logs = [np.log(p) for p in probs_list if p is not None]
            combined_log = np.mean(np.stack(logs, axis=0), axis=0)

            scaled = combined_log / float(self.temperature)
            exps = np.exp(scaled - np.max(scaled))
            combined_probs = exps / np.sum(exps)

            self.pred_history.append(combined_probs)
            smooth = np.mean(np.stack(self.pred_history, axis=0), axis=0)
            return smooth
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def predict_top_k(self, image_array: np.ndarray, k: int = 5) -> list[Dict[str, Any]]:
        try:
            predictions = self.predict(image_array)
            k = min(k, self.num_classes)
            top_indices = np.argsort(predictions)[-k:][::-1]
            results = []
            for rank, idx in enumerate(top_indices, 1):
                results.append({
                    'rank': rank,
                    'class': self.class_labels[idx],
                    'confidence': float(predictions[idx]),
                    'confidence_percent': float(predictions[idx] * 100)
                })
            return results
        except Exception as e:
            logger.error(f"Error getting top-k predictions: {e}")
            raise

    def predict_from_base64(self, base64_str: str, top_k: int = 5) -> list[Dict[str, Any]]:
        img_array = self.preprocess_base64(base64_str)
        return self.predict_top_k(img_array, k=top_k)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'input_name': self.input_name,
            'classes': self.class_labels,
            'providers': self.session.get_providers(),
            'model_path': self.model_path
        }
