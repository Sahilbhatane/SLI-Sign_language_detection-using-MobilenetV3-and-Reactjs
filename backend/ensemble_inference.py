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
from typing import Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ~20% padding on hand axis-aligned bounding box (from 21 keypoints)
HAND_BBOX_PAD_FRAC = 0.2


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

    def __init__(self, model_path: str = None, labels_path: str = None, enable_mediapipe_crop: bool = True, smoothing_window: int = 5):
        # Only initialize once
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Set default paths
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "model_v2.onnx")
        if labels_path is None:
            labels_path = os.path.join(os.path.dirname(__file__), "class_labels.txt")

        self.model_path = model_path
        self.labels_path = labels_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        logger.info(f"Loading ONNX model from: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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

        _env = os.environ.get("ENABLE_MEDIAPIPE_CROP", "1").strip()
        self.enable_mediapipe_crop = bool(enable_mediapipe_crop) and _env != "0"
        self._mp_hands = None
        self.last_hand_detected: bool = True
        self.last_wrist_norm_landmarks: Optional[np.ndarray] = None
        self._last_mp_raw_landmark_vec: Optional[np.ndarray] = None
        need_mediapipe = self.enable_mediapipe_crop or (self.landmark_session is not None)
        if need_mediapipe:
            try:
                import mediapipe as mp
                self._mp = mp
                # Single long-lived instance: real-time hand tracking, one hand (matches training focus)
                self._mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=0,
                )
                if self.enable_mediapipe_crop:
                    logger.info("MediaPipe hand detection + crop enabled (static_image_mode=False, max_num_hands=1)")
                if self.landmark_session is not None:
                    logger.info("MediaPipe landmark vectors enabled (landmark model loaded)")
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe: {e}")
                self._mp_hands = None
                self.enable_mediapipe_crop = False

        self.temperature = 1.0
        self._load_temperature()

        self.pred_history = deque(maxlen=max(1, int(smoothing_window)))

        logger.info("✓ Model loaded successfully")
        logger.info(f"  - Input shape: {self.input_shape}")
        logger.info(f"  - Target size for preprocessing: {self.target_size}")
        logger.info(f"  - Classes: {self.num_classes}")
        logger.info(f"  - Provider: {self.session.get_providers()[0]}")

        self._initialized = True

    def _extract_landmarks(self, image: Image.Image) -> Optional[np.ndarray]:
        """Run MediaPipe on a PIL image and return 63-dim (x,y,z)*21 vector (optional legacy path)."""
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

    def preprocess_image(self, image: Image.Image, target_size: tuple = None) -> np.ndarray:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            self.last_wrist_norm_landmarks = None
            self._last_mp_raw_landmark_vec = None
            ts = target_size or self.target_size or (224, 224)
            h_out, w_out = int(ts[0]), int(ts[1])

            if not (self.enable_mediapipe_crop and self._mp_hands is not None):
                self.last_hand_detected = True
            else:
                try:
                    image_np = np.asarray(image, dtype=np.uint8)
                    if image_np.ndim != 3 or image_np.shape[2] != 3:
                        self.last_hand_detected = True
                    else:
                        H, W, _ = image_np.shape
                        results = self._mp_hands.process(image_np)
                        if not results.multi_hand_landmarks:
                            self.last_hand_detected = False
                            img_std = np.full((h_out, w_out, 3), -1.0, dtype=np.float32)
                            return np.expand_dims(img_std, axis=0)
                        hand_lms = results.multi_hand_landmarks[0]
                        lms = list(hand_lms.landmark)
                        self.last_wrist_norm_landmarks = wrist_relative_hand_landmarks(lms)
                        self._last_mp_raw_landmark_vec = np.array(
                            [v for lm in lms for v in (lm.x, lm.y, lm.z)], dtype=np.float32
                        )
                        box = hand_bbox_from_landmarks(lms, W, H, HAND_BBOX_PAD_FRAC)
                        if box is None:
                            self.last_hand_detected = False
                            img_std = np.full((h_out, w_out, 3), -1.0, dtype=np.float32)
                            return np.expand_dims(img_std, axis=0)
                        x1, y1, x2, y2 = box
                        crop = image_np[y1:y2, x1:x2]
                        if crop.size == 0:
                            self.last_hand_detected = False
                            img_std = np.full((h_out, w_out, 3), -1.0, dtype=np.float32)
                            return np.expand_dims(img_std, axis=0)
                        image = Image.fromarray(crop)
                        self.last_hand_detected = True
                except Exception as e:
                    logger.debug(f"MediaPipe crop failed, using full image: {e}")
                    self.last_hand_detected = True

            # target_size is (H, W); PIL resize expects (W, H)
            image = image.resize((w_out, h_out), Image.Resampling.LANCZOS)

            img = np.array(image, dtype=np.float32)
            # Match ML/train_mobilenet.py: image / 127.5 - 1.0 -> [-1, 1]
            try:
                import tensorflow as tf  # optional; numpy path matches training numerics
                tensor = tf.convert_to_tensor(img)
                img_std = (tensor / 127.5 - 1.0).numpy()
            except Exception as e:
                logger.warning(f"TF normalize failed, using numpy fallback: {e}")
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
