# Training images (`data/`)

## Layout

- One **subfolder per class**; the folder name is the human-readable label (e.g. `stop`, `good morning`).
- Put RGB images inside each folder: `.png`, `.jpg`, `.jpeg`, or `.bmp`.
- [`ML/train_efficientnetv2.py`](../ML/train_efficientnetv2.py) (and the legacy [`ML/train_mobilenet.py`](../ML/train_mobilenet.py)) use `keras.utils.image_dataset_from_directory`, which **sorts class names alphabetically** and rewrites [`backend/class_labels.txt`](../backend/class_labels.txt) after every training run.

## Populating this directory

1. **Hugging Face (default):** sync the canonical phrase snapshot into this folder (same layout this repo trains on):
   ```bash
   hf auth whoami   # optional: confirm you are logged in
   python ML/pull_data_from_hf.py
   ```
   Use `SLI_HF_DATASET_REPO=namespace/name` to point at another dataset repo if needed.

2. **Download ISL phrase data from Mendeley (optional augmentation):**
   ```bash
   python ML/download_isl_phrases.py --dataset all
   ```
   Archives land under `datasets_raw/<key>/`.

3. **Merge into class folders:**
   ```bash
   python ML/ingest_external.py --mode local_images \
       --src datasets_raw/phrases_v2 \
       --mapping ML/external_gloss_mapping.example.yaml \
       --max-per-class 200
   ```
   Edit the mapping YAML so each source folder maps to the target `data/<class>` you want.

4. **Your own captures** — drop additional images into the right `data/<class>/` folder; the next training run picks them up automatically.

5. **Other corpora** — see [`../DATASETS.md`](../DATASETS.md) for Hugging Face and Kaggle sources, attribution, and license notes.

Maintainers refreshing the Hub copy after local merges: `python ML/upload_data_to_hf.py` (write token required).

## Class balance

`ML/train_efficientnetv2.py` prints per-class image counts, and when the max/min ratio is ≥ 3× it auto-enables `class_weight` in `model.fit` so rare classes are not ignored during training.

## Recovery when model files drift

If `backend/model_v2.onnx` and `backend/class_labels.txt` are out of sync with the folders in `data/`, predictions will map to the wrong labels. Recovery steps:

1. Re-sync `data/` (e.g. `python ML/pull_data_from_hf.py` or `python ML/ingest_external.py`).
2. Verify the expected class folders exist under `data/` and match your intended labels.
3. Retrain with `python ML/train_efficientnetv2.py` (this rewrites both `model_v2.onnx` and `class_labels.txt`).
4. Deploy `model_v2.onnx` and `class_labels.txt` together; never mix files from different training runs.

Quick check: the number of class folders under `data/` should match the number of lines in `backend/class_labels.txt`.

## Git

Image files under `data/` are **git-ignored** to keep the repository small. Only this `README.md` is tracked (`!data/README.md` in the root `.gitignore`).
