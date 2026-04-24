# Sign Language Recognition (Indian Sign Language — phrase level)

Real-time Indian Sign Language phrase detection with a FastAPI backend, a React + Tailwind frontend, and an EfficientNetV2-S model (with MobileNetV3-Large kept as a legacy option).

---

## Quick Start

**Windows (menu-driven):**

```batch
run.bat
```

Typical first run:

1. `1` — install Python backend dependencies.
2. `2` — install frontend dependencies.
3. `18` — download ISL phrase datasets from Mendeley.
4. `19` — merge downloaded images into `data/`.
5. `4` — train EfficientNetV2-S (uses GPU if CUDA is set up).
6. `7` — start FastAPI (`http://localhost:8000`).
7. `9` — start React dev server (`http://localhost:3000`).

**Manual setup:**

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..

# (Optional) bring in Indian Sign Language phrase data
python ML/download_isl_phrases.py --dataset all
python ML/ingest_external.py --mode local_images \
    --src datasets_raw/phrases_v2 \
    --mapping ML/external_gloss_mapping.example.yaml \
    --max-per-class 200

# Train (GPU auto-detected; mixed precision on CUDA)
python ML/train_efficientnetv2.py

# Serve
python backend/main.py                # terminal A
cd frontend && npm run dev            # terminal B
```

**URLs**

- Frontend: `http://localhost:3000`
- Backend:  `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

---

## Model

**Current (recommended):** `EfficientNetV2-S` via `tf.keras.applications.EfficientNetV2S`.

- ImageNet pretrained, ~20M params, 224 × 224 RGB.
- `include_preprocessing=False` with `[-1, 1]` normalization — **identical contract** to the prior MobileNet pipeline, so [`backend/ensemble_inference.py`](backend/ensemble_inference.py) does not change.
- Two-phase training: head-only, then top ~30% fine-tune.
- Mixed precision (`mixed_float16`) is enabled automatically when a GPU is visible, giving a large speed-up on RTX-class cards.
- Output files go to `backend/`: `best_model.h5`, `model_v2.onnx`, `class_labels.txt`, `training_history.png`.

**Why EfficientNetV2-S over MobileNetV3-Large?** Independent benchmarks show higher transfer-learning accuracy on small datasets (~0.2–0.5% average gain) with faster training on GPU thanks to fused MBConv blocks and better hardware utilization; EfficientNetV2-S is the author’s “sweet spot” for this category.

**Legacy:** `ML/train_mobilenet.py` (MobileNetV3-Large) remains available via menu option `17` and is useful when targeting very constrained hardware.

---

## Data

Images live under `data/<class_name>/…`. See [`data/README.md`](data/README.md) for layout rules. The directory itself is git-ignored so large image sets stay out of the repository.

Supported primary sources (phrase-level Indian Sign Language, CC BY 4.0):

- Mendeley `w7fgy7jvs8` v2 — 44 phrase classes × 40 images (matches the existing `data/` layout).
- Mendeley `y8vg69brn2` — 40 common ISL phrases × 30 images.

Secondary sources (ASL / word-level) and licensing notes live in [`DATASETS.md`](DATASETS.md). Download and ingest workflow is documented in [`ML/download_dataset.md`](ML/download_dataset.md).

---

## Project Structure

```
SLI/
├── backend/                    # FastAPI server
│   ├── main.py                 # API endpoints
│   ├── ensemble_inference.py   # ONNX inference + preprocessing
│   ├── model_v2.onnx           # Trained model (generated)
│   └── class_labels.txt        # Generated from data/ folders during training
├── frontend/                   # React app (Vite + Tailwind)
├── ML/
│   ├── train_efficientnetv2.py # Recommended trainer (EfficientNetV2-S, GPU)
│   ├── train_mobilenet.py      # Legacy trainer (MobileNetV3-Large)
│   ├── evaluate_model.py       # Offline metrics + confusion matrix
│   ├── ingest_external.py      # Merge HF / Kaggle / local media into data/
│   ├── download_isl_phrases.py # Pull Mendeley ISL phrase datasets
│   ├── download_dataset.md     # Step-by-step download & ingest commands
│   ├── fixtures/               # Tiny synthetic dataset for CI/preflight
│   └── inference.py            # Local ONNX inference utility
├── data/                       # Training images (git-ignored)
├── datasets_raw/               # Unpacked external archives (git-ignored)
├── DATASETS.md                 # Data sources, attribution, licensing
├── run.bat                     # Windows launcher (menu)
└── requirements.txt
```

---

## GPU training notes

- `ML/train_efficientnetv2.py` calls `tf.config.experimental.set_memory_growth` for each visible GPU and enables `mixed_float16` precision when a GPU is present.
- Confirm CUDA is available:
  ```bash
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  ```
  An empty list means TensorFlow cannot see the GPU — verify the CUDA/cuDNN versions match your TensorFlow wheel.
- Reduce `TrainConfig.BATCH_SIZE` from 32 → 16 if you hit OOM; input remains 224 × 224 for backend compatibility.

---

## Features

- Real-time webcam detection
- Phrase-level ISL / ASL sign classes (see `backend/class_labels.txt`; regenerated on every training run to match `data/`).
- Multi-language translation (9 languages)
- Optional browser voice output for stable detections
- Detection history tracking
- FastAPI backend with REST API (`/predict` accepts `min_confidence` 0–1)
- React frontend with TailwindCSS

---

## Testing

Frontend unit tests:

```bash
cd frontend && npm test
```

Preprocessing normalization parity (training <-> inference):

```bash
python ML/test_preprocess_norm.py
```

Training-pipeline preflight (uses `ML/fixtures/minimal_dataset` by default):

```bash
set SLI_VALIDATION_DATA_DIR=ML\fixtures\minimal_dataset
python ML/test_train_validation.py
python -m unittest ML.test_dataset_weights -v
```

---

## Troubleshooting

**Port in use:**
```bash
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

**Model not found:**
```bash
python ML/train_efficientnetv2.py
```

**GPU not detected:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
Reinstall CUDA/cuDNN versions that match your installed TensorFlow.

**Dependencies error:**
```bash
pip install -r requirements.txt --force-reinstall
```

**`sklearn` errors in `evaluate_model.py` after changing classes:**
`backend/class_labels.txt` must list the exact classes the current ONNX model predicts, in the same order. After adding folders under `data/`, retrain before evaluating.

---

## Resources

- **API Docs**: `http://localhost:8000/docs`
- **TensorFlow**: https://www.tensorflow.org
- **FastAPI**: https://fastapi.tiangolo.com
- **EfficientNetV2-S**: https://keras.io/api/applications/efficientnet_v2
