# External datasets (attribution and usage)

Use these sources to **expand** images under [`data/`](data/README.md). Always comply with each dataset’s **license** and **citation** requirements before redistributing models or data derived from them.

## Canonical training snapshot (this repository)

| Source | Size | Format | License | Use in this project |
|--------|------|--------|----------|---------------------|
| [Hugging Face `SahilBhatane/sli`](https://huggingface.co/datasets/SahilBhatane/sli) | Phrase-level class folders at the dataset repo root (mirrors local `data/<class>/`) | PNG / JPG | Bundle CC BY 4.0 sources (see dataset card) | **Default way to populate `data/`** on a fresh clone: `python ML/pull_data_from_hf.py` or `run.bat` option `3`. Override with `SLI_HF_DATASET_REPO` for a fork or private mirror. |

## Primary sources — Indian Sign Language, phrase-level

| Source | Size | Format | License | Use in this project |
|--------|------|--------|----------|---------------------|
| [Mendeley w7fgy7jvs8 v2](https://data.mendeley.com/datasets/w7fgy7jvs8/2) | 44 classes × 40 images | **PNG** (680×480) | CC BY 4.0 | **Primary upstream image pack.** Folder names match `data/` 1:1; ingest via the identity mapping helper below when building or refreshing the Hub snapshot. |
| [Mendeley y8vg69brn2](https://data.mendeley.com/datasets/y8vg69brn2/1) | 41 classes × ~900 arrays each | **`.npy` MediaPipe landmark arrays** | CC BY 4.0 | **Not** usable with the EfficientNetV2 image classifier. Useful only if you add a separate landmark-based classifier head. |
| [Hugging Face `akshaybahadur21/ISLAR`](https://huggingface.co/datasets/akshaybahadur21/ISLAR) | ~30k images | JPG/PNG | MIT | Gated — accept terms on the dataset page first, then set `HF_TOKEN`. |

Download the Mendeley datasets (API-backed; falls back to manual steps if Mendeley rotates URLs):

```bash
python ML/download_isl_phrases.py --dataset phrases_v2     # 44-class image set (recommended)
python ML/download_isl_phrases.py --dataset common_phrases # landmark .npy only (skip for image CNN)
```

Then build an identity mapping and merge into `data/`:

```bash
python ML/gen_identity_mapping.py \
    "datasets_raw/phrases_v2/images for phrases" \
    datasets_raw/phrases_v2_mapping.yaml

python ML/ingest_external.py --mode local_images \
    --src "datasets_raw/phrases_v2/images for phrases" \
    --mapping datasets_raw/phrases_v2_mapping.yaml \
    --max-per-class 400
```

See [`ML/download_dataset.md`](ML/download_dataset.md) for Kaggle / HF ingest patterns.

## Secondary / non-phrase references

| Source | Type | Fit for this project |
|--------|------|----------------------|
| [WLASL processed (Kaggle)](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) | Word-level ASL videos | Gloss → folder via `ingest_external.py` frame extraction. |
| [MS-ASL–style packs (Kaggle)](https://www.kaggle.com/datasets/saurabhshahane/american-sign-language-dataset) | ASL videos + JSON | Same. |
| [akasheroor/American-Sign-Language-Dataset (HF)](https://huggingface.co/datasets/akasheroor/American-Sign-Language-Dataset) | Large ASL video/word set | `--mode hf`. |
| [CISLR (HF)](https://huggingface.co/datasets/Exploration-Lab/CISLR) | Large ISL **video** corpus (~4,700 words) | Video → frames; verbose vocabulary. |
| [Voxel51/American-Sign-Language-MNIST (HF)](https://huggingface.co/datasets/Voxel51/American-Sign-Language-MNIST) | Static letters | Only if you add a separate letter task. |

## Credentials (never commit)

- **Kaggle:** set `KAGGLE_USERNAME` and `KAGGLE_KEY` (or use `kaggle.json` in the standard location). See [Kaggle API](https://www.kaggle.com/docs/api).
- **Hugging Face:** for gated datasets, `huggingface-cli login` or the `HF_TOKEN` environment variable. See [HF authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication).

## Ingestion commands

See [`ML/download_dataset.md`](ML/download_dataset.md) for example downloads. Run ingestion after placing archives under `datasets_raw/` or pointing `--src` at an unpacked tree:

```bash
python ML/ingest_external.py --help
```

Use [`ML/external_gloss_mapping.example.yaml`](ML/external_gloss_mapping.example.yaml) as a template for gloss → folder name mapping.
