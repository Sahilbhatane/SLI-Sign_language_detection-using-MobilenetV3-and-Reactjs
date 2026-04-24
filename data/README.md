# Training images (`data/`)

## Layout

- One **subfolder per class**; the folder name is the human-readable label (e.g. `stop`, `good morning`).
- Put RGB images inside each folder: `.png`, `.jpg`, `.jpeg`, or `.bmp`.
- [`ML/train_efficientnetv2.py`](../ML/train_efficientnetv2.py) (and the legacy [`ML/train_mobilenet.py`](../ML/train_mobilenet.py)) use `keras.utils.image_dataset_from_directory`, which **sorts class names alphabetically** and rewrites [`backend/class_labels.txt`](../backend/class_labels.txt) after every training run.

## Populating this directory

1. **Download ISL phrase data (recommended):**
   ```bash
   python ML/download_isl_phrases.py --dataset all
   ```
   Archives land under `datasets_raw/<key>/`.

2. **Merge into class folders:**
   ```bash
   python ML/ingest_external.py --mode local_images \
       --src datasets_raw/phrases_v2 \
       --mapping ML/external_gloss_mapping.example.yaml \
       --max-per-class 200
   ```
   Edit the mapping YAML so each source folder maps to the target `data/<class>` you want.

3. **Your own captures** — just drop additional images into the right `data/<class>/` folder; the next training run picks them up automatically.

4. **Other corpora** — see [`../DATASETS.md`](../DATASETS.md) for Hugging Face and Kaggle sources, attribution, and license notes.

## Class balance

`ML/train_efficientnetv2.py` prints per-class image counts, and when the max/min ratio is ≥ 3× it auto-enables `class_weight` in `model.fit` so rare classes are not ignored during training.

## Git

Image files under `data/` are **git-ignored** to keep the repository small. Only this `README.md` is tracked (`!data/README.md` in the root `.gitignore`).
