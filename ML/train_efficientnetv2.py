"""
Train EfficientNetV2-S for Indian Sign Language (or ASL) phrase classification.

Why this script exists:
- EfficientNetV2-S outperforms MobileNetV3-Large on transfer learning (small datasets)
  while training efficiently on GPU. See keras.io/api/applications/efficientnet_v2.
- Input is [-1, 1] with include_preprocessing=False, which matches the existing
  backend and MobileNet pipeline -> no changes needed in backend/ensemble_inference.py.
- Saves best_model.h5 + model_v2.onnx + class_labels.txt into backend/, so the
  FastAPI service picks up the new weights with no code changes.

Usage:
  python ML/train_efficientnetv2.py
"""

from __future__ import annotations

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '1')

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as exc:
        print(f"GPU configuration: {exc}")

tf.keras.backend.set_image_data_format('channels_last')

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import tf2onnx


# ==========================
# Configuration
# ==========================
class Config:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = str((ROOT_DIR / "data").resolve())
    BACKEND_DIR = str((ROOT_DIR / "backend").resolve())

    @classmethod
    def ensure_backend_dir(cls):
        os.makedirs(cls.BACKEND_DIR, exist_ok=True)


class TrainConfig:
    # Keep 224x224 to stay drop-in compatible with backend's current 224x224 ONNX contract.
    # Optional upgrade: raise to 300 or 384 for a few more points of accuracy (retrain + re-export).
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32  # Fits on 6GB GPUs with mixed precision; reduce to 16 if OOM.
    EPOCHS = 40
    FINE_TUNE_EPOCHS = 25
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 1e-3
    FINE_TUNE_LR = 1e-5  # Lower than head LR; EfficientNetV2 base weights are sensitive.
    LABEL_SMOOTHING = 0.1

    USE_MIXED_PRECISION = True  # Large speed-up on CUDA-capable GPUs.

    MODEL_H5_PATH = os.path.join(Config.BACKEND_DIR, "best_model.h5")
    MODEL_ONNX_PATH = os.path.join(Config.BACKEND_DIR, "model_v2.onnx")
    TRAINING_HISTORY_PATH = os.path.join(Config.BACKEND_DIR, "training_history.png")


def maybe_enable_mixed_precision() -> str:
    if TrainConfig.USE_MIXED_PRECISION and gpus:
        try:
            keras.mixed_precision.set_global_policy('mixed_float16')
            return "mixed_float16"
        except Exception as exc:  # pragma: no cover
            print(f"mixed precision disabled: {exc}")
    return "float32"


# ==========================
# Data augmentation
# ==========================
def _build_augmenter():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.12),
        layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
        layers.RandomZoom(height_factor=0.10, width_factor=0.10),
        layers.RandomContrast(factor=0.2),
        layers.RandomBrightness(factor=0.15),
    ], name="augmentation")


def _preprocess_fn(training: bool):
    """[-1, 1] normalization (same contract as backend inference)."""
    augmenter = _build_augmenter()

    @tf.function
    def fn(image, label):
        image = tf.cast(image, tf.float32)
        if training:
            image = augmenter(image, training=True)
        image = image / 127.5 - 1.0
        return image, label

    return fn


def create_datasets():
    data_dir = Config.DATA_DIR
    img_size = (TrainConfig.IMG_HEIGHT, TrainConfig.IMG_WIDTH)

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=TrainConfig.VALIDATION_SPLIT,
        subset='training',
        seed=42,
        image_size=img_size,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=True,
        color_mode='rgb',
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=TrainConfig.VALIDATION_SPLIT,
        subset='validation',
        seed=42,
        image_size=img_size,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=False,
        color_mode='rgb',
    )
    class_names = train_ds.class_names

    train_ds = train_ds.map(_preprocess_fn(training=True), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_preprocess_fn(training=False), num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE), class_names


# ==========================
# Class balance helpers (shared semantics with train_mobilenet.py)
# ==========================
def count_images_per_class(data_dir: str, class_names: list) -> list:
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}
    root = Path(data_dir)
    counts: list = []
    for name in class_names:
        d = root / name
        if not d.is_dir():
            counts.append(0)
            continue
        seen = set()
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in valid_ext:
                seen.add(p.resolve())
        counts.append(len(seen))
    return counts


def compute_class_weight_if_imbalanced(counts: list, ratio_threshold: float = 3.0):
    positive = [c for c in counts if c > 0]
    if len(positive) < 2:
        return None
    ma, mi = max(positive), min(positive)
    if mi == 0 or ma / mi < ratio_threshold:
        return None
    total = float(sum(counts))
    n = len(counts)
    return {i: total / (n * max(1, counts[i])) for i in range(n)}


# ==========================
# Model
# ==========================
def build_model(num_classes: int):
    print("Building EfficientNetV2-S with ImageNet weights...")
    base = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(TrainConfig.IMG_HEIGHT, TrainConfig.IMG_WIDTH, 3),
        include_preprocessing=False,  # We normalize to [-1, 1] ourselves.
    )
    base.trainable = False
    print("EfficientNetV2-S loaded.")

    inputs = layers.Input(shape=(TrainConfig.IMG_HEIGHT, TrainConfig.IMG_WIDTH, 3), name='input')
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    # Softmax in float32 for numerical stability under mixed precision.
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions', dtype='float32')(x)
    return keras.Model(inputs, outputs, name='EfficientNetV2S_SignLanguage'), base


def get_callbacks():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(Config.BACKEND_DIR, f'logs/effv2_{timestamp}')
    return [
        EarlyStopping(monitor='val_accuracy', patience=15, min_delta=0.001, restore_best_weights=True, mode='max', verbose=1),
        ModelCheckpoint(TrainConfig.MODEL_H5_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False, update_freq='epoch', profile_batch=0),
    ]


def top_3_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def plot_history(history1, history2, save_path):
    hist = {
        'accuracy': history1.history.get('accuracy', []) + history2.history.get('accuracy', []),
        'val_accuracy': history1.history.get('val_accuracy', []) + history2.history.get('val_accuracy', []),
        'loss': history1.history.get('loss', []) + history2.history.get('loss', []),
        'val_loss': history1.history.get('val_loss', []) + history2.history.get('val_loss', []),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('EfficientNetV2-S Training History', fontsize=16, fontweight='bold')
    ax1.plot(hist['accuracy'], label='train acc')
    ax1.plot(hist['val_accuracy'], label='val acc')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(hist['loss'], label='train loss')
    ax2.plot(hist['val_loss'], label='val loss')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("EfficientNetV2-S Training - Sign Language Phrases")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")
    policy = maybe_enable_mixed_precision()
    print(f"Precision policy: {policy}")
    print(f"Batch size: {TrainConfig.BATCH_SIZE}  | Image size: {TrainConfig.IMG_HEIGHT}x{TrainConfig.IMG_WIDTH}")
    print(f"Data: {Config.DATA_DIR}")

    Config.ensure_backend_dir()

    print("\n[1/5] Loading datasets...")
    train_ds, val_ds, class_names = create_datasets()
    num_classes = len(class_names)
    counts = count_images_per_class(Config.DATA_DIR, class_names)
    total = sum(counts)
    split = TrainConfig.VALIDATION_SPLIT
    train_est = int(round(total * (1.0 - split)))
    val_est = max(0, total - train_est)
    class_weight = compute_class_weight_if_imbalanced(counts)
    print(f"Classes: {num_classes}")
    if counts:
        print(f"Images per class (min/median/max): {min(counts)} / {int(np.median(counts))} / {max(counts)}")
    print(f"Total images: {total}  | ~train/val: {train_est}/{val_est}")
    print("class_weight:", "enabled" if class_weight is not None else "disabled")

    class_labels_path = os.path.join(Config.BACKEND_DIR, 'class_labels.txt')
    with open(class_labels_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))
    print(f"Class labels written to: {class_labels_path}")

    print("\n[2/5] Building model...")
    model, base = build_model(num_classes)
    model.summary()

    loss = keras.losses.CategoricalCrossentropy(label_smoothing=TrainConfig.LABEL_SMOOTHING)

    print("\n[3/5] Phase 1: head-only training...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TrainConfig.LEARNING_RATE),
        loss=loss,
        metrics=['accuracy', top_3_accuracy],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TrainConfig.EPOCHS,
        callbacks=get_callbacks(),
        class_weight=class_weight,
        verbose=1,
    )

    print("\n[4/5] Phase 2: fine-tuning (top ~30% of base)...")
    base.trainable = True
    freeze_until = int(len(base.layers) * 0.7)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TrainConfig.FINE_TUNE_LR),
        loss=loss,
        metrics=['accuracy', top_3_accuracy],
    )
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TrainConfig.FINE_TUNE_EPOCHS,
        callbacks=get_callbacks(),
        class_weight=class_weight,
        verbose=1,
    )

    print("\n[5/5] Evaluation + export...")
    val_loss, val_acc, val_top3 = model.evaluate(val_ds, verbose=0)
    print(f"val_accuracy: {val_acc*100:.2f}%  top-3: {val_top3*100:.2f}%  loss: {val_loss:.4f}")

    plot_history(history, history_ft, TrainConfig.TRAINING_HISTORY_PATH)
    print(f"Training history plot: {TrainConfig.TRAINING_HISTORY_PATH}")

    try:
        spec = tf.TensorSpec((None, TrainConfig.IMG_HEIGHT, TrainConfig.IMG_WIDTH, 3), tf.float32, name="input")
        tf2onnx.convert.from_keras(model, input_signature=[spec], opset=13, output_path=TrainConfig.MODEL_ONNX_PATH)
        print(f"ONNX export: {TrainConfig.MODEL_ONNX_PATH}")
    except Exception as exc:
        print(f"ONNX export failed: {exc}")

    log_dir = Path(Config.BACKEND_DIR) / 'logs'
    if log_dir.exists():
        try:
            shutil.rmtree(log_dir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
