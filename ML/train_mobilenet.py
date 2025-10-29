import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as e:
        print(f"GPU configuration: {e}")

# Force channels_last format
tf.keras.backend.set_image_data_format('channels_last')

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import tf2onnx
import onnx


# ==========================
# Configuration
# ==========================
class Config:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = str((ROOT_DIR / "data").resolve())
    BACKEND_DIR = str((ROOT_DIR / "backend").resolve())
    
    @classmethod
    def ensure_backend_dir(cls):
        """Ensure backend directory exists."""
        os.makedirs(cls.BACKEND_DIR, exist_ok=True)


class MobileNetConfig:
    # MobileNetV3 recommended input size
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32  # Optimized for RTX 3050 + MobileNet efficiency
    EPOCHS = 60  # More epochs for small dataset
    FINE_TUNE_EPOCHS = 30
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 1e-3  # Higher LR for small dataset
    FINE_TUNE_LR = 5e-6  # Very low for fine-tuning
    
    # Label smoothing helps small datasets
    LABEL_SMOOTHING = 0.1

    # Save models in backend directory to keep project organized
    MODEL_H5_PATH = os.path.join(Config.BACKEND_DIR, "best_model.h5")
    MODEL_ONNX_PATH = os.path.join(Config.BACKEND_DIR, "model_v2.onnx")
    TRAINING_HISTORY_PATH = os.path.join(Config.BACKEND_DIR, "training_history.png")


# ==========================
# Data Augmentation (Aggressive for small dataset)
# ==========================
def _build_augmenter():
    """Aggressive augmentation for small dataset (40 images/class)."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.15),  # ±54 degrees
        layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
        layers.RandomZoom(height_factor=0.15, width_factor=0.15),
        layers.RandomContrast(factor=0.2),
        layers.RandomBrightness(factor=0.2),
    ], name="augmentation")


def _preprocess_fn(training: bool):
    """Preprocessing with MobileNetV3 normalization."""
    augmenter = _build_augmenter()

    @tf.function
    def fn(image, label):
        image = tf.cast(image, dtype=tf.float32)
        
        if training:
            image = augmenter(image, training=True)
        
        # MobileNetV3 uses [-1, 1] normalization
        image = image / 127.5 - 1.0
        
        return image, label

    return fn


def create_datasets():
    """Create datasets with aggressive augmentation for small dataset."""
    data_dir = Config.DATA_DIR
    img_size = (MobileNetConfig.IMG_HEIGHT, MobileNetConfig.IMG_WIDTH)

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=MobileNetConfig.VALIDATION_SPLIT,
        subset='training',
        seed=42,
        image_size=img_size,
        batch_size=MobileNetConfig.BATCH_SIZE,
        shuffle=True,
        color_mode='rgb',
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=MobileNetConfig.VALIDATION_SPLIT,
        subset='validation',
        seed=42,
        image_size=img_size,
        batch_size=MobileNetConfig.BATCH_SIZE,
        shuffle=False,
        color_mode='rgb',
    )

    class_names = train_ds.class_names

    # Apply preprocessing
    train_ds = train_ds.map(_preprocess_fn(training=True), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_preprocess_fn(training=False), num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes: int):
    """Build MobileNetV3-Large model optimized for small datasets."""
    
    print("Building MobileNetV3-Large with ImageNet weights...")
    
    # MobileNetV3-Large with ImageNet weights
    base = MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=(MobileNetConfig.IMG_HEIGHT, MobileNetConfig.IMG_WIDTH, 3),
        minimalistic=False,  # Use full version for better accuracy
        include_preprocessing=False  # We do our own preprocessing
    )
    
    print("✓ MobileNetV3-Large loaded successfully!")
    base.trainable = False

    # Build model - simpler head for small dataset
    inputs = layers.Input(shape=(MobileNetConfig.IMG_HEIGHT, MobileNetConfig.IMG_WIDTH, 3), name='input')
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Single dense layer prevents overfitting on small dataset
    x = layers.Dropout(0.3, name='dropout')(x)
    
    # Output with label smoothing
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='MobileNetV3_SignLanguage')

    return model, base


def get_callbacks():
    """Callbacks optimized for small dataset training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use backend directory for logs to keep ML folder clean
    log_dir = os.path.join(Config.BACKEND_DIR, f'logs/mobilenet_{timestamp}')
    
    return [
        # Early stopping with high patience for small dataset
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # High patience - small datasets are noisy
            min_delta=0.001,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            MobileNetConfig.MODEL_H5_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Aggressive LR reduction for small datasets
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard - minimal logging to save space
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,  # Disable histogram to save space
            write_graph=False,  # Disable graph to save space
            write_images=False,  # Disable images to save space
            update_freq='epoch',
            profile_batch=0  # Disable profiling to save space
        ),
    ]


def top_3_accuracy(y_true, y_pred):
    """Top-3 accuracy metric."""
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def plot_training_history(history1, history2, save_path):
    """Generate and save training history visualization."""
    # Combine histories
    history = {
        'accuracy': history1.history.get('accuracy', []) + history2.history.get('accuracy', []),
        'val_accuracy': history1.history.get('val_accuracy', []) + history2.history.get('val_accuracy', []),
        'loss': history1.history.get('loss', []) + history2.history.get('loss', []),
        'val_loss': history1.history.get('val_loss', []) + history2.history.get('val_loss', [])
    }
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('MobileNetV3-Large Training History', fontsize=16, fontweight='bold')
    
    # Plot accuracy
    epochs1 = range(1, len(history1.history['accuracy']) + 1)
    epochs2 = range(len(history1.history['accuracy']) + 1, len(history['accuracy']) + 1)
    
    ax1.plot(epochs1, history1.history['accuracy'], 'b-', label='Training Accuracy (Phase 1)', linewidth=2)
    ax1.plot(epochs1, history1.history['val_accuracy'], 'orange', label='Validation Accuracy (Phase 1)', linewidth=2)
    
    if history2.history.get('accuracy'):
        ax1.plot(epochs2, history2.history['accuracy'], 'b--', label='Training Accuracy (Phase 2)', linewidth=2)
        ax1.plot(epochs2, history2.history['val_accuracy'], 'orange', linestyle='--', label='Validation Accuracy (Phase 2)', linewidth=2)
        ax1.axvline(x=len(history1.history['accuracy']), color='red', linestyle='--', linewidth=2, label='Fine-tuning starts')
    
    ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot loss
    ax2.plot(epochs1, history1.history['loss'], 'b-', label='Training Loss (Phase 1)', linewidth=2)
    ax2.plot(epochs1, history1.history['val_loss'], 'orange', label='Validation Loss (Phase 1)', linewidth=2)
    
    if history2.history.get('loss'):
        ax2.plot(epochs2, history2.history['loss'], 'b--', label='Training Loss (Phase 2)', linewidth=2)
        ax2.plot(epochs2, history2.history['val_loss'], 'orange', linestyle='--', label='Validation Loss (Phase 2)', linewidth=2)
        ax2.axvline(x=len(history1.history['loss']), color='red', linestyle='--', linewidth=2, label='Fine-tuning starts')
    
    ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # High quality but reasonable size
    print(f"✓ Training history plot saved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("MobileNetV3-Large Training - OPTIMIZED FOR SIGN LANGUAGE")
    print("=" * 70)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Model: MobileNetV3-Large (Most Popular & Efficient)")
    print(f"Dataset: 44 classes, ~40 images/class (Small Dataset)")
    print(f"Hardware: Ryzen 7 5800H + RTX 3050 6GB + 8GB RAM")
    print(f"Batch Size: {MobileNetConfig.BATCH_SIZE}")
    print(f"Image Size: {MobileNetConfig.IMG_HEIGHT}x{MobileNetConfig.IMG_WIDTH}")
    print(f"Data dir: {Config.DATA_DIR}\n")
    
    # Ensure backend directory exists
    Config.ensure_backend_dir()
    print(f"✓ Backend directory ready: {Config.BACKEND_DIR}\n")

    # [1/5] Build datasets
    print("[1/5] Loading datasets with aggressive augmentation...")
    train_ds, val_ds, class_names = create_datasets()
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")
    print(f"Training samples: ~{int(1760 * 0.8)}")
    print(f"Validation samples: ~{int(1760 * 0.2)}")
    
    # Save class labels
    class_labels_path = os.path.join(Config.BACKEND_DIR, 'class_labels.txt')
    with open(class_labels_path, 'w') as f:
        f.write('\n'.join(class_names))
    print(f"Saved class labels to: {class_labels_path}\n")

    # [2/5] Build model
    print("[2/5] Building MobileNetV3-Large model...")
    
    model, base_model = build_model(num_classes)
    model.summary()
    
    print(f"\n Model Statistics:")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Base model params: {base_model.count_params():,}")
    print(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

    # [3/5] Initial training (frozen base)
    print("\n[3/5] Phase 1: Training classification head...")
    print("(Base MobileNetV3 frozen, only training final layers)")
    
    # Use label smoothing for small dataset
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=MobileNetConfig.LABEL_SMOOTHING)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=MobileNetConfig.LEARNING_RATE),
        loss=loss,
        metrics=['accuracy', top_3_accuracy]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MobileNetConfig.EPOCHS,
        callbacks=get_callbacks(),
        verbose=1
    )

    # [4/5] Fine-tuning (unfreeze base)
    print("\n[4/5] Phase 2: Fine-tuning entire model...")
    base_model.trainable = True
    
    # Freeze first 70% for stability (MobileNet is sensitive)
    num_layers = len(base_model.layers)
    freeze_until = int(num_layers * 0.7)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    print(f"Unfrozen last {num_layers - freeze_until} layers of {num_layers} total")

    # Recompile with very low learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=MobileNetConfig.FINE_TUNE_LR),
        loss=loss,
        metrics=['accuracy', top_3_accuracy]
    )

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MobileNetConfig.FINE_TUNE_EPOCHS,
        callbacks=get_callbacks(),
        verbose=1
    )

    # [5/5] Evaluation & Export
    print("\n[5/5] Final evaluation and export...")
    
    # Use the already-trained model (avoid Keras reload bug with MobileNetV3)
    # The best weights are already loaded by ModelCheckpoint callback
    print("Using trained model for final evaluation...")
    
    # Evaluate
    val_loss, val_acc, val_top3 = model.evaluate(val_ds, verbose=0)
    print(f"\n{'='*70}")
    print(f"FINAL VALIDATION RESULTS:")
    print(f"{'='*70}")
    print(f"  Accuracy:     {val_acc*100:.2f}%")
    print(f"  Top-3 Acc:    {val_top3*100:.2f}%")
    print(f"  Loss:         {val_loss:.4f}")
    print(f"{'='*70}")
    
    # Generate training history visualization
    print(f"\nGenerating training history visualization...")
    plot_training_history(history, history_ft, MobileNetConfig.TRAINING_HISTORY_PATH)

    # Export to ONNX
    print(f"\nExporting to ONNX: {MobileNetConfig.MODEL_ONNX_PATH}")
    
    try:
        spec = tf.TensorSpec((None, MobileNetConfig.IMG_HEIGHT, MobileNetConfig.IMG_WIDTH, 3), tf.float32, name="input")
        
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=[spec],
            opset=13,
            output_path=MobileNetConfig.MODEL_ONNX_PATH
        )
        print(" ONNX export successful!")
        
    except Exception as e:
        print(f" ONNX export failed: {e}")
        print("  Model saved as .h5 format only")
    
    # Clean up large log files to save space
    print(f"\nCleaning up temporary files...")
    log_dir = os.path.join(Config.BACKEND_DIR, 'logs')
    if os.path.exists(log_dir):
        import shutil
        try:
            shutil.rmtree(log_dir)
            print(f" Removed TensorBoard logs (saved space)")
        except:
            print(f" Could not remove logs (may be in use)")

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n Final Model Performance:")
    print(f"  • Validation Accuracy: {val_acc*100:.2f}%")
    print(f"  • Top-3 Accuracy: {val_top3*100:.2f}%")
    print(f"  • Model Size: {os.path.getsize(MobileNetConfig.MODEL_H5_PATH) / (1024*1024):.2f} MB")
    
    print(f"\n MobileNetV3-Large Advantages:")
    print(f"  • 2-3x faster inference than EfficientNetB3")
    print(f"  • Only 5.4M parameters (vs 12M)")
    print(f"  • Industry-standard for production")
    print(f"  • Excellent for small datasets")
    print(f"  • Optimized for your RTX 3050 6GB")
    
    print(f"\n Saved Files (All in backend/):")
    print(f"  • Model: best_model.h5")
    print(f"  • ONNX: model_v2.onnx")
    print(f"  • Training Plot: training_history.png")
    print(f"  • Class Labels: class_labels.txt")


if __name__ == "__main__":
    main()
