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
import tensorflow as tf

logger = logging.getLogger(__name__)


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

    def __init__(self, model_path: str = None, labels_path: str = None, enable_mediapipe_crop: bool = False, smoothing_window: int = 5):
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

        self.enable_mediapipe_crop = enable_mediapipe_crop or os.environ.get("ENABLE_MEDIAPIPE_CROP", "0") == "1"
        self._mp_hands = None
        need_mediapipe = self.enable_mediapipe_crop or (self.landmark_session is not None)
        if need_mediapipe:
            try:
                import mediapipe as mp
                self._mp = mp
                self._mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    model_complexity=1,
                )
                if self.enable_mediapipe_crop:
                    logger.info("MediaPipe hand cropping enabled")
                if self.landmark_session is not None:
                    logger.info("MediaPipe landmark extraction enabled (landmark model loaded)")
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
        if self._mp_hands is None:
            return None
        try:
            img_np = np.array(image)
            H, W = img_np.shape[0], img_np.shape[1]
            results = self._mp_hands.process(img_np)
            if not results.multi_hand_landmarks:
                return None
            best = None
            best_area = -1
            for hand_lms in results.multi_hand_landmarks:
                xs = [lm.x * W for lm in hand_lms.landmark]
                ys = [lm.y * H for lm in hand_lms.landmark]
                x1, x2 = max(0, int(min(xs))), min(W, int(max(xs)))
                y1, y2 = max(0, int(min(ys))), min(H, int(max(ys)))
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = hand_lms
            if best is None:
                return None
            pts = []
            for lm in best.landmark:
                pts.extend([lm.x, lm.y, lm.z])
            vec = np.array(pts, dtype=np.float32)
            return vec
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

            if self.enable_mediapipe_crop and self._mp_hands is not None:
                try:
                    image_np = np.array(image)
                    results = self._mp_hands.process(image_np)
                    if results.multi_hand_landmarks:
                        H, W, _ = image_np.shape
                        boxes = []
                        for hand_lms in results.multi_hand_landmarks:
                            xs = [lm.x * W for lm in hand_lms.landmark]
                            ys = [lm.y * H for lm in hand_lms.landmark]
                            x1, x2 = max(0, int(min(xs))), min(W, int(max(xs)))
                            y1, y2 = max(0, int(min(ys))), min(H, int(max(ys)))
                            boxes.append((x1, y1, x2, y2))
                        if boxes:
                            x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                            mx = int(0.1 * (x2 - x1 + 1))
                            my = int(0.1 * (y2 - y1 + 1))
                            x1 = max(0, x1 - mx)
                            y1 = max(0, y1 - my)
                            x2 = min(W, x2 + mx)
                            y2 = min(H, y2 + my)
                            image = Image.fromarray(image_np[y1:y2, x1:x2])
                except Exception as e:
                    logger.debug(f"MediaPipe crop failed, using full image: {e}")

            ts = target_size or self.target_size or (224, 224)
            image = image.resize(ts, Image.Resampling.LANCZOS)

            img = np.array(image, dtype=np.float32)
            try:
                tensor = tf.convert_to_tensor(img)
                tensor = tf.image.per_image_standardization(tensor)
                img_std = tensor.numpy()
            except Exception as e:
                logger.warning(f"TF standardization failed, using numpy fallback: {e}")
                m = np.mean(img, dtype=np.float32)
                s = np.std(img, dtype=np.float32)
                s = float(max(s, 1.0/np.sqrt(img.size)))
                img_std = (img - m) / s

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
            probs_list = []

            outputs = self.session.run(None, {self.input_name: image_array})
            raw_img = outputs[0][0]
            probs_list.append(np.clip(raw_img, 1e-8, 1.0))

            if self.landmark_session is not None:
                try:
                    H, W = image_array.shape[1], image_array.shape[2]
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
