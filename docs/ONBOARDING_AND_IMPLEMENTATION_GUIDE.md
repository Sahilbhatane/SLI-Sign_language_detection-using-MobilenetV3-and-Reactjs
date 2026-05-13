# SLI onboarding and implementation guide

This document is for **new developers** and **new machines** with no prior context on the Sign Language Interpreter (SLI) project. It summarizes what the assistive stack does, what was added on branch **`TTS/VLM`**, how to run it end-to-end, and how to verify behavior with tests and manual checks.

For a shorter reference (env vars, API links), see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

---

## 1. What this project is

| Area | Path / tech | Role |
|------|----------------|------|
| UI | `frontend/` — React, Vite, Tailwind | Webcam, detection panels, translation, voice (TTS), sentence builder, settings |
| API | `backend/` — FastAPI, ONNX Runtime | `/predict`, health, optional `/translate`, `/tts`, `/llm/correct`, `/fallback`, optional WebRTC `/ws/webrtc` |
| Model training | `ML/` | Train EfficientNetV2-S (recommended) or legacy MobileNet; exports `backend/model_v2.onnx` + `class_labels.txt` |

The **Detect** tab drives real-time inference: either **REST** (JPEG snapshots to `/predict`) or **WebRTC** (optional: browser sends video; server runs inference on sampled frames). The UI tries WebRTC when you start detection and falls back to REST if signaling or the connection fails or drops.

---

## 2. What was implemented (TTS/VLM and integration pass)

Below is a concise list of **behavior and files** you should know about when reading or changing code.

### 2.1 WebRTC vs REST (frontend + backend)

- **`frontend/src/services/webrtcClient.ts`** — WebSocket to `/ws/webrtc`, SDP offer/answer, `RTCPeerConnection`, prediction messages on a dedicated listener after the answer; **`connectionstatechange`**, **`iceconnectionstatechange`**, and WS **close** trigger **`onFallback`** and **`close()`** to avoid leaks.
- **`frontend/src/components/CameraPanel.jsx`** — On **Start detection**, calls **`startWebRtcSession`**. On success: **`transport = webrtc`**, REST polling disabled. On failure or runtime fallback: **`transport = rest`**, REST interval resumes.
- **`frontend/src/components/WebcamCapture.jsx`** — Props **`useRestPolling`**, **`onCapturingChange`**, **`onFpsSample`**. REST **`/api/predict`** loop runs only when **`useRestPolling`** is true.
- **`frontend/vite.config.js`** — Proxies **`/api`** and **`/ws`** to `http://localhost:8000` in dev.
- **`backend/webrtc_server.py`** (optional, requires **`aiortc`**) — Registers **`/ws/webrtc`**: applies the browser’s offer, **`createAnswer`**, sends answer, runs inference on received video frames (stride from **`WEBRTC_FRAME_STRIDE`**).

### 2.2 TTS, sentence pipeline, LLM, translation

- **`frontend/src/services/ttsService.ts`** — Confidence gate (**≥ 0.95**) for phrase-level speech, cooldown, and **semantic dedupe** (default **8s**): the same phrase is not spoken again within that window even if the short cooldown expired.
- **`frontend/src/hooks/useSentencePipeline.ts`** + **`frontend/src/hooks/sentencePipeline.js`** — Phrase must be stable for **N** frames (**5**) with **average confidence ≥ 0.97** over those frames (and each frame **≥ 0.95** via streak rules). Sentence finalize: **speaks raw (or translated) text first**; **LLM correction** runs in the **background** and may trigger a **second** speak only when overlap with the raw sentence is in a safe band (see `tokenJaccardSimilarity` in `sentencePipeline.js`).
- **`backend/extra_routes.py`** — **`POST /llm/correct`**: after OpenAI/Ollama, applies **token Jaccard similarity ≥ 0.7**, **length cap**, and **extra-sentence** guard; otherwise returns the **original** text.
- **`frontend/src/services/translationService.ts`** — Order: **local gloss** (`frontend/public/gloss/isl_gloss.json`, keys normalized: lowercase, snake_case, compact) → **`/api/translate`** → **original text** if the API fails.

### 2.3 UI and observability

- **`frontend/src/components/DebugObservability.jsx`** — Shown only when **`import.meta.env.DEV`**: transport (WebRTC/REST), approximate FPS (REST path), last detection confidence, TTS provider, LLM grammar on/off and optional follow-up TTS flag.
- **`frontend/src/App.jsx`** — Owns **`transport`**, wires **`CameraPanel`**, sentence hook **`onLlmFollowUp`** for the debug strip.

### 2.4 Tests added or extended

- **`frontend/src/services/ttsService.test.ts`** — Cooldown + semantic dedupe + server fallback.
- **`frontend/src/hooks/sentencePipeline.test.ts`** — Streak, average confidence gate, Jaccard helper.
- **`frontend/src/services/translationService.test.ts`** — Gloss key variants.
- **`backend/test_llm_sanitize.py`** — **`unittest`**: Jaccard and **`_sanitize_llm_correction`** behavior (run from `backend/`).
- **`backend/test_integration_api.py`** — **`unittest`**: FastAPI integration for core endpoints using a fake model (no ONNX needed).

---

## 3. Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.10+** (recommended) | Same interpreter for backend and optional WebRTC extras |
| **Node.js 18+** and **npm** | For `frontend/` |
| **Git** | Clone the repo; active feature branch is **`TTS/VLM`** |
| **ONNX model** | `backend/model_v2.onnx` + `backend/class_labels.txt` — from training (`ML/train_efficientnetv2.py`) or copied from a teammate |

**Recommended:** create a virtual environment at the repo root so `pip` does not touch system Python:

```powershell
cd C:\path\to\SLI
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. How to run everything

### 4.1 Windows menu (`run.bat`)

From the **repository root**:

```batch
run.bat
```

Typical sequence for **first-time developers**:

| Step | Menu option | What it does |
|------|-------------|----------------|
| 1 | **1** | `pip install -r requirements.txt` (use venv’s `pip` if you created `venv`) |
| 2 | **2** | `npm install` in `frontend/` |
| 3 | **16** (optional) | `pip install -r requirements-webrtc.txt` — enables **`/ws/webrtc`** when the API starts |
| 4 | **4** (if you have no model) | Train EfficientNetV2-S → writes `backend/model_v2.onnx`, etc. |
| 5 | **7** | Start FastAPI: **`http://localhost:8000`**, docs at **`/docs`** |
| 6 | **9** | Start Vite dev server: **`http://localhost:3000`** (proxies `/api` and `/ws` to port 8000) |

Keep **7** and **9** in **two terminals** (or two `run.bat` windows). Open the app at **`http://localhost:3000`**, go to **Detect**, start the camera, then **Start detection**.

### 4.2 Manual commands (same as CI or Linux-style shells)

**Terminal A — backend**

```powershell
cd C:\path\to\SLI
.\venv\Scripts\activate   # if using venv
pip install -r requirements.txt
# Optional WebRTC:
pip install -r requirements-webrtc.txt
python backend\main.py
```

**Terminal B — frontend**

```powershell
cd C:\path\to\SLI\frontend
npm install
npm run dev
```

Optional WebRTC deps are listed in **`requirements-webrtc.txt`** ( **`aiortc`** ). Without them, the server logs that WebRTC is disabled; the UI still works on **REST** only.

### 4.3 Frontend-only launcher

`frontend/run-frontend.bat` can install deps, run dev, build, or preview. The UI still needs the API on **port 8000** for predictions unless you only care about static assets.

### 4.4 Environment configuration

Copy and edit examples if present (`backend/.env.example`, `frontend/.env`). Important variables are summarized in [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) ( **`ALLOWED_ORIGINS`**, translation URL, TTS, LLM keys, **`WEBRTC_*`**, **`VITE_STUN_URLS`** ).

---

## 5. How to test everything

### 5.1 Automated — frontend (Vitest)

From **`frontend/`**:

```powershell
cd frontend
npm test
```

Or CI-style once:

```powershell
npm run test -- --run
```

**Covers:** `ttsService`, `sentencePipeline` helpers, `translationService` gloss keys, `signSpeech` utilities (see `src/**/*.test.*`).

**Production build smoke test:**

```powershell
npm run build
```

### 5.2 Automated — backend (LLM sanitize unit tests)

From **`backend/`** (repo root venv should have `requirements.txt` installed; tests only import `extra_routes`):

```powershell
cd backend
..\venv\Scripts\python.exe -m unittest test_llm_sanitize -v
```

### 5.3 Automated — backend integration (FastAPI, fake model)

From **`backend/`**:

```powershell
cd backend
..\venv\Scripts\python.exe -m unittest test_integration_api -v
```

**Covers:** `/`, `/health`, `/classes`, `/model-info`, and `/predict` without loading the ONNX model.

### 5.4 Manual / integration — API script

With the **server running** on port 8000:

```powershell
cd C:\path\to\SLI
python backend\test_api.py
```

`run.bat` option **8** runs the same script (it expects a live server).

### 5.5 Manual — UI checklist (Detect tab)

| # | Action | Expected |
|---|--------|----------|
| 1 | Open **`http://localhost:3000`**, Detect tab | Backend indicator shows connected when API is up |
| 2 | Start camera + **Start detection** | Predictions update; transport shows **webrtc** if optional deps + server path work, else **rest** |
| 3 | Stop detection | Transport returns to **rest**; no duplicate intervals after toggling |
| 4 | Non-English language + translation | Panel shows translation; failures fall back to phrase or original per `translationService` |
| 5 | Voice mode on | Phrase TTS only when confidence **≥ 0.95**; repeated phrase not re-spoken inside semantic window |
| 6 | Hold stable sign ~5 frames with high confidence | Phrase enters sentence buffer; idle **~3s** triggers sentence flow (raw speak first; optional LLM follow-up if enabled) |
| 7 | Settings — LLM grammar | With backend keys, `/llm/correct` improves text; unsafe drift is rejected server-side |
| 8 | Dev build (`npm run dev`) | Debug overlay (bottom-right): transport, FPS, confidence, TTS, LLM flags |

### 5.6 WebRTC-specific checks

1. Install **`requirements-webrtc.txt`**, restart **`python backend/main.py`**.
2. Confirm no log line saying WebRTC is disabled.
3. In the UI, start detection: **transport** should show **webrtc** when negotiation succeeds.
4. Disconnect network or stop server: UI should **fall back to REST** when the client reports failure/disconnect (and after restart, REST should work without duplicate loops).

---

## 6. Where to read code next

| Topic | Start here |
|-------|------------|
| Prediction entry to sentence buffer | `useSentencePipeline.ts`, `sentencePipeline.js` |
| REST capture loop | `WebcamCapture.jsx` |
| WebRTC orchestration | `CameraPanel.jsx`, `webrtcClient.ts` |
| HTTP API surface | `backend/extra_routes.py`, `backend/main.py` |
| ONNX predict path | Search for `predict` / `model_v2` in `backend/` |

---

## 7. Troubleshooting (quick)

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| WebRTC always **rest** | `aiortc` not installed or WS error | Install option **16** / `requirements-webrtc.txt`; check browser console and server logs |
| **`/tts` 501** | No gTTS / ElevenLabs configured | Use **Edge** / browser TTS in UI, or set env per `DEVELOPER_GUIDE.md` |
| Import errors for `onnxruntime` | Wrong Python / no venv | Use the same venv you used for `pip install -r requirements.txt` |
| Blank predictions | Missing or wrong ONNX / labels | Train or copy `model_v2.onnx` + `class_labels.txt` into `backend/` |

---

*Last updated for branch **TTS/VLM** (assistive stack: FastAPI + React + ONNX + optional WebRTC + TTS + LLM + translation).*
