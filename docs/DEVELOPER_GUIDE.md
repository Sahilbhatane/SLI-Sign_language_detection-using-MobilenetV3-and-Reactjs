# SLI developer guide (branch `TTS/VLM`)

**New to the repo?** Start with **[`ONBOARDING_AND_IMPLEMENTATION_GUIDE.md`](ONBOARDING_AND_IMPLEMENTATION_GUIDE.md)** — full stack overview, what was built on this branch, how to run and test on a clean machine.

This file is a **compact reference**: URLs, env vars, and quick test bullets.

---

This repository is a full-stack sign-language assistant:

- `frontend/` — React + Vite + Tailwind
- `backend/` — FastAPI + ONNX inference
- `ML/` — training + dataset utilities
- `docs/` — documentation (this file + onboarding guide)

OpenAPI (when the API is running): `http://localhost:8000/docs`

## Quick start (Windows)

Use **`run.bat`** from the repo root (see onboarding doc for the full menu walkthrough):

1. **Install backend deps**: option **1** (prefer a venv: `python -m venv venv` then `.\venv\Scripts\pip install -r requirements.txt`)
2. **Install frontend deps**: option **2**
3. **Optional WebRTC**: option **16** — `pip install -r requirements-webrtc.txt`
4. **Start API**: option **7** (needs `backend/model_v2.onnx` + `backend/class_labels.txt` from training option **4**)
5. **Start UI**: option **9** → `http://localhost:3000` (Vite proxies `/api` and `/ws` to port 8000)

Manual commands:

```powershell
.\venv\Scripts\pip install -r requirements.txt
pip install -r requirements-webrtc.txt
cd frontend
npm install
npm run dev
```

In another terminal:

```powershell
.\venv\Scripts\python.exe backend\main.py
```

## Branch

Feature work for the assistive upgrade is on **`TTS/VLM`**.

## Environment variables

### Backend (`backend/.env` or process env)

| Name | Purpose |
|------|---------|
| `ALLOWED_ORIGINS` | Comma-separated origins, or `*` |
| `LIBRETRANSLATE_URL` | Upstream LibreTranslate-compatible base URL |
| `ENABLE_GTTS` | `1` enables optional `gTTS` for `/tts` |
| `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | Optional ElevenLabs for `/tts` |
| `OPENAI_API_KEY` | Enables `/llm/correct` (OpenAI) and optional `/fallback` vision |
| `OPENAI_LLM_MODEL`, `OPENAI_VLM_MODEL` | Model names (defaults in `.env.example` if present) |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Optional local LLM for `/llm/correct` + `/fallback` text path |
| `WEBRTC_FRAME_STRIDE`, `WEBRTC_MIN_CONFIDENCE` | WebRTC inference sampling / gating |

### Frontend (`frontend/.env`)

| Name | Purpose |
|------|---------|
| `VITE_STUN_URLS` | Comma-separated STUN URLs for WebRTC |
| `VITE_ENABLE_INDIAN_EXTRA` | `1` adds Tamil/Telugu to the language selector |

## Primary user flows to test

1. **Health**: `GET /api/health` from the UI indicator or curl.
2. **REST predict**: start detection with WebRTC unavailable or after fallback; predictions update on the **~250ms** capture loop.
3. **WebRTC** (optional): install `requirements-webrtc.txt`, restart API; start detection — UI should show **transport: webrtc** when negotiation succeeds; disconnect falls back to **rest**.
4. **Translation**: choose Hindi/Marathi; `/api/translate` proxy; local **`public/gloss/isl_gloss.json`** overrides when keys match (normalized variants).
5. **Voice mode**: TTS uses confidence **≥ 0.95** for phrase path; **semantic dedupe** (~8s) suppresses repeated identical phrases after cooldown.
6. **Sentence pipeline**: **5** stable frames with **average confidence ≥ 0.97** (per-frame ≥ 0.95); idle **~3s** finalizes — **raw (or translated) spoken first**, optional **LLM follow-up** in background when grammar is enabled.
7. **LLM grammar** (optional): Settings toggle; `/llm/correct` rejects low token-overlap or over-long / extra-sentence outputs (returns original).
8. **Low-confidence fallback**: with `OPENAI_API_KEY` set, `/api/fallback` may return `used_fallback=true` for weak frames.
9. **Export transcript**: button on the detect screen (downloads JSON).
10. **Dev debug overlay**: visible in `npm run dev` only — transport, FPS (REST), last confidence, TTS provider, LLM flags.

## Implemented vs deferred fallback matrix

| Layer | Implemented in this branch | Deferred / harderening |
|------|----------------------------|-------------------------|
| Streaming | **REST** `/predict` + optional **WebRTC** `/ws/webrtc` (UI tries WebRTC first, REST fallback) | ICE trickle, production **TURN** |
| TTS | Browser + `/tts` (optional gTTS / ElevenLabs); semantic phrase dedupe | Coqui local server bundle |
| LLM | `/llm/correct` OpenAI→Ollama + **similarity / length / sentence guards** | Dedicated on-device grammar model |
| Translation | `/translate` proxy + **gloss-first** `isl_gloss.json` + **original text** on API failure | Google Cloud Translation proxy |
| Vision | `/fallback` OpenAI vision if key present | Gemini path |

## Security notes

- Never commit API keys. Use `.env` / CI secrets.
- Treat all client input as untrusted; backend routes validate lengths and use timeouts (`httpx`).
- Prefer `ALLOWED_ORIGINS` not `*` in production.

## Tests

**Frontend** (Vitest, from `frontend/`):

```powershell
cd frontend
npm test
npm run test -- --run
```

**Backend** (LLM sanitize helpers, from `backend/`):

```powershell
cd backend
..\venv\Scripts\python.exe -m unittest test_llm_sanitize -v
```

**Live API** (requires server on port 8000):

```powershell
python backend\test_api.py
```

## Troubleshooting

- **WebRTC disabled log** (`aiortc not available`): install `requirements-webrtc.txt` into the same venv used to run the API (`run.bat` option **16**).
- **`/tts` returns 501**: expected until `ENABLE_GTTS=1` or ElevenLabs env vars are configured; the UI falls back to browser TTS for `server`/`elevenlabs` provider paths.
- **TensorFlow logs on import**: benign; set `TF_CPP_MIN_LOG_LEVEL=3` to reduce noise.
- **ICE `disconnected` triggers REST fallback**: can be transient on poor networks; see `webrtcClient.ts` if you need to narrow to `failed` only.
