# Downloading external corpora

Do not commit API keys or `kaggle.json`. Use environment variables or host-level config.

## Indian Sign Language phrase datasets (primary)

Mendeley datasets are CC BY 4.0 and public. `download_isl_phrases.py` calls the Mendeley public API and downloads presigned S3 URLs; it prints manual steps if that ever fails.

```bash
python ML/download_isl_phrases.py --dataset phrases_v2       # 44 classes × 40 PNG images (use this)
python ML/download_isl_phrases.py --dataset common_phrases   # 41 classes × MediaPipe .npy arrays (NOT images)
```

`phrases_v2` extracts to `datasets_raw/phrases_v2/images for phrases/<class>/*.png` and its folder names already match the project's `data/<class>` layout.

Generate an identity mapping automatically and ingest:

```bash
python ML/gen_identity_mapping.py \
    "datasets_raw/phrases_v2/images for phrases" \
    datasets_raw/phrases_v2_mapping.yaml

python ML/ingest_external.py --mode local_images \
    --src "datasets_raw/phrases_v2/images for phrases" \
    --mapping datasets_raw/phrases_v2_mapping.yaml \
    --max-per-class 400
```

If you need to merge a dataset whose folder names differ from `data/<class>`, hand-write the mapping YAML instead, e.g.:

```yaml
# my_isl_map.yaml
STOP: stop
"good morning": good morning
HELLO: hello
```

If Mendeley changes URLs, the helper prints the browser download link. Open the page, click **Download All**, unpack into `datasets_raw/<key>/`, then re-run the ingest step.

**Note:** `common_phrases` (y8vg69brn2) contains `.npy` MediaPipe landmark arrays, not raw images — useful only if you add a separate landmark-based classifier head. It is not consumed by `ML/train_efficientnetv2.py`.

## Kaggle (WLASL-style example)

Install: `pip install kaggle`

```bash
mkdir -p datasets_raw
cd datasets_raw
kaggle datasets download -d risangbaskoro/wlasl-processed
# unzip manually or: tar -xf *.zip  (format varies)
```

Set `KAGGLE_USERNAME` and `KAGGLE_KEY` from your Kaggle account **Account → Create New API Token**.

## Hugging Face

Install: `pip install huggingface_hub datasets`

```bash
huggingface-cli download akasheroor/American-Sign-Language-Dataset --repo-type dataset --local-dir datasets_raw/asl-akasheroor
```

For gated datasets, run `huggingface-cli login` first or set `HF_TOKEN`.

## After download

1. Unpack archives under `datasets_raw/` (gitignored) or any local path.
2. Edit a copy of [`external_gloss_mapping.example.yaml`](external_gloss_mapping.example.yaml) so gloss names from the corpus map to your `data/<folder>` names.
3. Run [`ingest_external.py`](ingest_external.py); see `--help` for `--mode hf`, `local_videos`, and `local_images`.
