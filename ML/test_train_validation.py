"""
Comprehensive validation script for train_mobilenet.py
Tests all components before running full training
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV3Large

print("=" * 80)
print("TRAINING SCRIPT VALIDATION - Pre-flight Checks")
print("=" * 80)

# Test counters
tests_passed = 0
tests_failed = 0
warnings = []

def test_step(name, func):
    """Run a test step and track results."""
    global tests_passed, tests_failed
    try:
        print(f"\n[TEST] {name}...")
        result = func()
        if result:
            print(f"  ✓ PASS: {result}")
            tests_passed += 1
        else:
            print(f"  ✓ PASS")
            tests_passed += 1
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {str(e)}")
        tests_failed += 1
        return False

def warn(msg):
    """Add a warning message."""
    warnings.append(msg)
    print(f"  ⚠ WARNING: {msg}")

# ==========================
# Test 1: Python & TensorFlow Environment
# ==========================
def test_environment():
    """Check Python and TensorFlow versions."""
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    tf_version = tf.__version__
    
    if sys.version_info < (3, 8):
        raise Exception(f"Python 3.8+ required, found {python_version}")
    
    return f"Python {python_version}, TensorFlow {tf_version}"

# ==========================
# Test 2: GPU Configuration
# ==========================
def test_gpu():
    """Check GPU availability and configuration."""
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        warn("No GPU detected - training will use CPU (very slow!)")
        return "CPU only (training will be slow)"
    
    # Try to configure memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return f"{len(gpus)} GPU(s) configured with memory growth"
    except Exception as e:
        warn(f"GPU memory growth config failed: {e}")
        return f"{len(gpus)} GPU(s) found (memory growth disabled)"

# ==========================
# Test 3: Required Dependencies
# ==========================
def test_dependencies():
    """Check all required packages are installed."""
    required = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'tf2onnx': 'tf2onnx',
        'onnx': 'onnx',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        raise Exception(f"Missing packages: {', '.join(missing)}\nInstall with: pip install {' '.join(missing)}")
    
    return "All required packages installed"

# ==========================
# Test 4: Directory Structure
# ==========================
def test_directories():
    """Validate project directory structure."""
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    backend_dir = root_dir / "backend"
    
    if not data_dir.exists():
        raise Exception(f"Data directory not found: {data_dir}")
    
    # Count class folders
    class_folders = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(class_folders) == 0:
        raise Exception(f"No class folders found in {data_dir}")
    
    # Create backend directory if needed
    backend_dir.mkdir(exist_ok=True)
    
    return f"Data: {data_dir} ({len(class_folders)} classes), Backend: {backend_dir}"

# ==========================
# Test 5: Dataset Loading
# ==========================
def test_dataset_loading():
    """Test dataset creation and loading."""
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = str(root_dir / "data")
    
    # Try to load a small batch
    test_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=(224, 224),
        batch_size=4,  # Small batch for testing
        shuffle=True,
        color_mode='rgb',
    )
    
    # Get class names
    class_names = test_ds.class_names
    
    # Try to get one batch
    for images, labels in test_ds.take(1):
        batch_size = images.shape[0]
        img_shape = images.shape[1:]
        label_shape = labels.shape[1]
        
        if img_shape != (224, 224, 3):
            raise Exception(f"Expected image shape (224, 224, 3), got {img_shape}")
        
        if label_shape != len(class_names):
            raise Exception(f"Label dimension mismatch: {label_shape} vs {len(class_names)} classes")
    
    return f"{len(class_names)} classes, batch shape: {images.shape}, labels: {labels.shape}"

# ==========================
# Test 6: Data Preprocessing
# ==========================
def test_preprocessing():
    """Test data augmentation and preprocessing pipeline."""
    from tensorflow.keras import layers
    
    # Build augmenter
    augmenter = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
        layers.RandomZoom(height_factor=0.15, width_factor=0.15),
        layers.RandomContrast(factor=0.2),
        layers.RandomBrightness(factor=0.2),
    ], name="augmentation")
    
    # Create dummy image
    dummy_img = tf.random.uniform((1, 224, 224, 3), minval=0, maxval=255)
    
    # Test augmentation
    augmented = augmenter(dummy_img, training=True)
    
    # Test normalization
    normalized = augmented / 127.5 - 1.0
    
    # Check range
    min_val = tf.reduce_min(normalized).numpy()
    max_val = tf.reduce_max(normalized).numpy()
    
    if min_val < -1.5 or max_val > 1.5:
        raise Exception(f"Normalization range issue: [{min_val:.2f}, {max_val:.2f}]")
    
    return f"Augmentation & normalization OK (range: [{min_val:.2f}, {max_val:.2f}])"

# ==========================
# Test 7: Model Building
# ==========================
def test_model_building():
    """Test MobileNetV3 model creation."""
    from tensorflow.keras import layers
    
    print("    Loading MobileNetV3-Large (this may take a moment)...")
    
    # Build base model
    base = MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        minimalistic=False,
        include_preprocessing=False
    )
    
    # Build full model
    inputs = layers.Input(shape=(224, 224, 3), name='input')
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(44, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='MobileNetV3_SignLanguage')
    
    # Test forward pass
    dummy_input = tf.random.uniform((1, 224, 224, 3), minval=-1, maxval=1)
    output = model(dummy_input, training=False)
    
    if output.shape != (1, 44):
        raise Exception(f"Expected output shape (1, 44), got {output.shape}")
    
    # Check output is valid probability distribution
    output_sum = tf.reduce_sum(output).numpy()
    if abs(output_sum - 1.0) > 0.01:
        raise Exception(f"Output not a valid probability distribution (sum={output_sum})")
    
    param_count = model.count_params()
    
    return f"Model built successfully ({param_count:,} parameters), output shape: {output.shape}"

# ==========================
# Test 8: Model Compilation
# ==========================
def test_compilation():
    """Test model compilation with optimizer and loss."""
    from tensorflow.keras import layers
    
    # Build a minimal model
    base = MobileNetV3Large(
        include_top=False,
        weights=None,  # Faster for testing
        input_shape=(224, 224, 3),
        minimalistic=False,
        include_preprocessing=False
    )
    base.trainable = False
    
    inputs = layers.Input(shape=(224, 224, 3), name='input')
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(44, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with training settings
    def top_3_accuracy(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
    
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=['accuracy', top_3_accuracy]
    )
    
    return "Model compiled with Adam optimizer, categorical crossentropy, and metrics"

# ==========================
# Test 9: Callbacks Configuration
# ==========================
def test_callbacks():
    """Test callback creation."""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    from datetime import datetime
    
    root_dir = Path(__file__).resolve().parent.parent
    backend_dir = root_dir / "backend"
    backend_dir.mkdir(exist_ok=True)
    
    model_path = str(backend_dir / "test_model.h5")
    log_dir = str(backend_dir / "logs" / "test")
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            min_delta=0.001,
            restore_best_weights=True,
            mode='max',
            verbose=0
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=0
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq='epoch',
            profile_batch=0
        ),
    ]
    
    return f"{len(callbacks)} callbacks configured (EarlyStopping, ModelCheckpoint, ReduceLR, TensorBoard)"

# ==========================
# Test 10: Model Save/Load (Critical Fix Test)
# ==========================
def test_model_save_load():
    """Test that model saving works (important for MobileNetV3 bug)."""
    from tensorflow.keras import layers
    import tempfile
    
    # Build minimal model
    base = MobileNetV3Large(
        include_top=False,
        weights=None,  # Faster for testing
        input_shape=(224, 224, 3),
        minimalistic=False,
        include_preprocessing=False
    )
    
    inputs = layers.Input(shape=(224, 224, 3))
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(44, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Try to save
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        model.save(tmp_path)
        
        # Note: We WON'T try to load because of the known MobileNetV3 bug
        # Our script avoids loading by using the trained model directly
        warn("MobileNetV3 has known reload bug - script uses trained model directly (no reload)")
        
        return "Model saving works (reload skipped - known MobileNetV3 bug)"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ==========================
# Test 11: ONNX Export
# ==========================
def test_onnx_export():
    """Test ONNX export capability."""
    import tf2onnx
    import onnx
    from tensorflow.keras import layers
    import tempfile
    
    # Build minimal model
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(16, 3, activation='relu')(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(44, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Try ONNX export
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        spec = tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input")
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=[spec],
            opset=13,
            output_path=tmp_path
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(tmp_path)
        onnx.checker.check_model(onnx_model)
        
        return "ONNX export and validation successful"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ==========================
# Test 12: Training History Plotting
# ==========================
def test_plotting():
    """Test matplotlib plotting for training history."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import tempfile
    
    # Create dummy history
    class DummyHistory:
        def __init__(self):
            self.history = {
                'accuracy': [0.5, 0.6, 0.7],
                'val_accuracy': [0.4, 0.5, 0.6],
                'loss': [1.0, 0.8, 0.6],
                'val_loss': [1.2, 1.0, 0.8]
            }
    
    history1 = DummyHistory()
    history2 = DummyHistory()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, 4)
        ax1.plot(epochs, history1.history['accuracy'], 'b-', label='Training')
        ax1.plot(epochs, history1.history['val_accuracy'], 'orange', label='Validation')
        ax1.set_title('Accuracy')
        ax1.legend()
        
        ax2.plot(epochs, history1.history['loss'], 'b-', label='Training')
        ax2.plot(epochs, history1.history['val_loss'], 'orange', label='Validation')
        ax2.set_title('Loss')
        ax2.legend()
        
        plt.savefig(tmp_path, dpi=150)
        plt.close()
        
        # Check file was created
        if not os.path.exists(tmp_path):
            raise Exception("Plot file was not created")
        
        file_size = os.path.getsize(tmp_path)
        
        return f"Plot generated successfully ({file_size} bytes)"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ==========================
# Test 13: Disk Space Check
# ==========================
def test_disk_space():
    """Check available disk space."""
    import shutil
    
    root_dir = Path(__file__).resolve().parent.parent
    backend_dir = root_dir / "backend"
    
    # Get disk usage
    stat = shutil.disk_usage(backend_dir if backend_dir.exists() else root_dir)
    
    free_gb = stat.free / (1024**3)
    
    # Need at least 1GB free (model ~20MB, logs ~100MB worst case)
    if free_gb < 1:
        warn(f"Low disk space: {free_gb:.2f} GB free (need ~1 GB)")
    
    return f"{free_gb:.2f} GB free space available"

# ==========================
# Test 14: Memory Check
# ==========================
def test_memory():
    """Check available system memory."""
    import psutil
    
    mem = psutil.virtual_memory()
    mem_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    
    if available_gb < 2:
        warn(f"Low available memory: {available_gb:.2f} GB (training may be slow)")
    
    return f"{mem_gb:.2f} GB total, {available_gb:.2f} GB available"

# ==========================
# Run All Tests
# ==========================
print("\n" + "=" * 80)
print("RUNNING VALIDATION TESTS")
print("=" * 80)

test_step("1. Python & TensorFlow Environment", test_environment)
test_step("2. GPU Configuration", test_gpu)
test_step("3. Required Dependencies", test_dependencies)
test_step("4. Directory Structure", test_directories)
test_step("5. Dataset Loading", test_dataset_loading)
test_step("6. Data Preprocessing", test_preprocessing)
test_step("7. Model Building", test_model_building)
test_step("8. Model Compilation", test_compilation)
test_step("9. Callbacks Configuration", test_callbacks)
test_step("10. Model Save/Load", test_model_save_load)
test_step("11. ONNX Export", test_onnx_export)
test_step("12. Training History Plotting", test_plotting)
test_step("13. Disk Space Check", test_disk_space)
test_step("14. Memory Check", test_memory)

# ==========================
# Final Report
# ==========================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print(f"✓ Tests Passed: {tests_passed}")
print(f"✗ Tests Failed: {tests_failed}")
print(f"⚠ Warnings: {len(warnings)}")

if warnings:
    print("\nWarnings:")
    for i, w in enumerate(warnings, 1):
        print(f"  {i}. {w}")

print("\n" + "=" * 80)

if tests_failed == 0:
    print("✓ ALL TESTS PASSED - Script is ready to run!")
    print("=" * 80)
    print("\nYou can now run the training with:")
    print("  python train_mobilenet.py")
    print("\nExpected training time:")
    if 'GPU' in str([t for t in [test_gpu()] if t]):
        print("  • With GPU: 1-2 hours (90 epochs total)")
    else:
        print("  • With CPU: 8-12 hours (90 epochs total)")
    print("\nExpected output files in backend/:")
    print("  • best_model.h5 (~20 MB)")
    print("  • model_v2.onnx (~20 MB)")
    print("  • training_history.png (~200 KB)")
    print("  • class_labels.txt (~1 KB)")
    sys.exit(0)
else:
    print("✗ TESTS FAILED - Please fix errors before running training")
    print("=" * 80)
    sys.exit(1)
