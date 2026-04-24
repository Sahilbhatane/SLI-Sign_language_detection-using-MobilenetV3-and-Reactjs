Here is a single **from-scratch runbook** you can follow on a **new Windows machine** (nothing installed yet), assuming the **code is on GitHub** and the **phrase images are on Hugging Face** (`SahilBhatane/sli`), in an order that avoids the usual footguns.

---

## 0. What “fully running” means

You need **two processes**: the **Python/FastAPI backend** (loads the ONNX model) and the **React frontend** (webcam UI).  
**“Accurate detection”** means: **same classes and preprocessing as the model**. Easiest path:

- **Fast path (no training):** use the `**backend/model_v2.onnx`** + `**backend/class_labels.txt**` already in the repo *if* they match the Hub dataset (they should if you shipped them together).
- **Best path (retrain on this machine):** after syncing `data/` from HF, run **EfficientNetV2-S** training once on your GPU so weights match your exact local snapshot.

---

## 1. Install base software (once per machine)

1. **Git** — [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. **Python 3.11.x (64-bit)** — [https://www.python.org/downloads/](https://www.python.org/downloads/)
  - During install: enable **“Add python.exe to PATH”**.
3. **Node.js LTS** (includes `npm`) — [https://nodejs.org/](https://nodejs.org/)

Optional but recommended for **GPU training / faster TF**:

1. **NVIDIA GPU driver** + **CUDA/cuDNN stack that matches your TensorFlow wheel** (see TensorFlow install docs for your exact TF version). If you skip this, the project can still run on **CPU**; training will be slow.

---

## 2. Get the code

```powershell
cd $HOME\Desktop
git clone https://github.com/Sahilbhatane/SLI-Sign_language_detection-using-MobilenetV3-and-Reactjs.git SLI
cd SLI
```

(Use your real clone URL if it differs.)

---

## 3. Python environment (backend + ML)

```powershell
cd SLI
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**If `Activate.ps1` is blocked** (execution policy):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 4. Hugging Face CLI + auth (for dataset sync)

Still in the **activated venv**:

```powershell
hf auth whoami
```

- If that errors or shows logged out:  
  ```powershell
  hf auth login
  ```
  Use a token with **read** access to the dataset (public dataset: read token or fine-grained read is enough).

**Default dataset repo** is `SahilBhatane/sli`. To use another repo:

```powershell
set SLI_HF_DATASET_REPO=YourNamespace/your-dataset-repo
```

---

## 5. Pull training images into `data/` (required before training)

From repo root, venv active:

```powershell
python ML\pull_data_from_hf.py
```

Or use `**run.bat` → option `3**`.

This fills `**data/<class>/...**` (gitignored locally).  
**Do not** expect training images inside Git; they come from HF by design.

---

## 6. Choose inference path

### 6A. Quick run (no training) — works if repo ONNX matches labels

Check these exist:

- `backend\model_v2.onnx`
- `backend\class_labels.txt`

If both exist and you **did not** change classes under `data/` vs what the ONNX was trained on, you can **skip training** and go straight to serving (section 8).  
If you later add/remove class folders under `data/`, you **must** retrain (6B) or you will get wrong or unstable predictions.

### 6B. Full accuracy path (recommended after fresh sync) — train once

```powershell
python ML\train_efficientnetv2.py
```

This regenerates `**backend/model_v2.onnx**`, `**backend/class_labels.txt**`, plots, etc., aligned with **your** `data/` tree.

**GPU check (optional):**

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Empty list ⇒ TF is CPU-only (still works; training slower).

---

## 7. Frontend dependencies

New terminal **or** deactivate venv; from repo root:

```powershell
cd frontend
npm install
cd ..
```

---

## 8. Run the app (two terminals)

**Terminal A — backend**

```powershell
cd SLI
.\venv\Scripts\Activate.ps1
python backend\main.py
```

Backend: **[http://localhost:8000](http://localhost:8000)** — open **[http://localhost:8000/docs](http://localhost:8000/docs)** to confirm it’s up.

**Terminal B — frontend**

```powershell
cd SLI\frontend
npm run dev
```

Frontend: **[http://localhost:3000](http://localhost:3000)** (or the port Vite prints).

Allow **camera** in the browser when prompted.

---

## 9. Menu-driven alternative (Windows)

From `SLI`:

```powershell
.\run.bat
```

Typical sequence:

1. `1` — install backend deps (if you didn’t use venv + pip already).
2. `2` — frontend `npm install` (if not done).
3. `3` — sync `data/` from Hugging Face.
4. `4` — train EfficientNetV2-S (for best accuracy on this snapshot).
5. `7` — start API (if `run.bat` starts server for you; otherwise still use `python backend\main.py` as above).
6. `9` — start React dev server.

(Exact menu labels match your current `run.bat`.)

---

## 10. “No errors” checklist (things that break new installs)


| Symptom                                  | Fix                                                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `python` / `pip` not found               | Reinstall Python with **Add to PATH**, reopen terminal.                                          |
| `ModuleNotFoundError` (tensorflow, etc.) | `.\venv\Scripts\Activate.ps1` then `pip install -r requirements.txt`.                            |
| Empty `data/` / training says no classes | Run `**python ML\pull_data_from_hf.py`** (or `run.bat` `3`) **after** `hf auth login` if needed. |
| `403` / auth errors pulling dataset      | `hf auth login` or set `**HF_TOKEN`**; for private forks set `**SLI_HF_DATASET_REPO**`.          |
| Model missing / wrong classes            | Run `**python ML\train_efficientnetv2.py**` after `data/` is complete.                           |
| Port 8000 busy                           | Stop other apps or change backend port in `backend/main.py` / config if you added one.           |
| Webcam blocked                           | Browser site permissions → allow camera for `localhost`.                                         |
| GPU not used                             | Driver + CUDA/cuDNN match TF wheel; otherwise CPU still runs.                                    |


---

## 11. Accuracy (what actually matters)

1. `**data/**` matches the **phrase classes** you care about (from HF snapshot).
2. `**backend/class_labels.txt`** order matches the **trained softmax** (regenerated by training).
3. **Lighting + framing** — model is image-based; poor crop or motion hurts accuracy.
4. After any **ingest** / folder rename, **retrain** so ONNX and labels stay in sync.

---

## 12. Optional sanity commands (before demo)

```powershell
python ML\test_preprocess_norm.py
python ML\test_pull_data_from_hf.py
```

(Training stack tests like `ML\test_dataset_weights.py` import TensorFlow; run them only after TF installs cleanly.)

---

### Shortest “happy path” summary

**New PC:** Install Git, Python 3.11, Node → clone → `python -m venv venv` → activate → `pip install -r requirements.txt` → `hf auth login` → `python ML\pull_data_from_hf.py` → `python ML\train_efficientnetv2.py` (recommended) → `cd frontend && npm install` → Terminal A: `python backend\main.py` → Terminal B: `npm run dev` in `frontend` → open **localhost:3000**, allow camera.

That sequence matches how the repo is designed today (Git for code + small artifacts, Hugging Face for images, local `data/` for training and optional retrain for maximum accuracy).