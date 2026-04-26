# SLI developer guide (branch `TTS/VLM`)

This repository is a full-stack sign-language assistant:

- `frontend/` — React + Vite + Tailwind
- `backend/` — FastAPI + ONNX inference
- `ML/` — training + dataset utilities
- `docs/` — documentation (this file)

OpenAPI (when the API is running): `http://localhost:8000/docs`

## Quick start (Windows)

Use `run.bat`:

1. **Install backend deps**: option **1**
2. **Install frontend deps**: option **2**
3. **Start API**: option **7** (needs `backend/model_v2.onnx` + `backend/class_labels.txt` from training option **4**)
4. **Start UI**: option **9**

Manual commands:

```powershell
.\venv\Scripts\pip install -r requirements.txt
cd frontend
npm install
npm run dev
```

In another terminal:

```powershell
.\venv\Scripts\python.exe backend\main.py
```

## Branch

Feature work for this upgrade is on **`TTS/VLM`**.

## Environment variables

### Backend (`backend/.env` or process env)

| Name | Purpose |
|------|---------|
| `ALLOWED_ORIGINS` | Comma-separated origins, or `*` |
| `LIBRETRANSLATE_URL` | Upstream LibreTranslate-compatible base URL |
| `ENABLE_GTTS` | `1` enables optional `gTTS` for `/tts` |
| `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | Optional ElevenLabs for `/tts` |
| `OPENAI_API_KEY` | Enables `/llm/correct` (OpenAI) and optional `/fallback` vision |
| `OPENAI_LLM_MODEL`, `OPENAI_VLM_MODEL` | Model names (defaults in `.env.example`) |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Optional local LLM for `/llm/correct` + `/fallback` text path |
| `WEBRTC_FRAME_STRIDE`, `WEBRTC_MIN_CONFIDENCE` | WebRTC inference sampling / gating |

### Frontend (`frontend/.env`)

| Name | Purpose |
|------|---------|
| `VITE_STUN_URLS` | Comma-separated STUN URLs for WebRTC |
| `VITE_ENABLE_INDIAN_EXTRA` | `1` adds Tamil/Telugu to the language selector |

## Primary user flows to test

1. **Health**: `GET /api/health` from the UI indicator or curl.
2. **REST predict**: start detection; confirm predictions update ~4×/s (250ms loop).
3. **Translation**: choose Hindi/Marathi; confirm translation panel updates (uses `/api/translate` proxy).
4. **Voice mode**: enable Voice Mode; confirm TTS only fires when model confidence is **≥ 0.95** (see `ttsService.ts`).
5. **Sentence pipeline**: hold a stable sign for **5** matching frames; confirm phrase enters buffer; stop for **3s**; confirm “Forming sentence” cue then final speech.
6. **LLM grammar** (optional): enable in Settings; requires backend LLM configuration; otherwise server returns original text.
7. **Low-confidence fallback**: with `OPENAI_API_KEY` set, `/api/fallback` may return `used_fallback=true` for weak frames.
8. **WebRTC** (optional): install `requirements-webrtc.txt`, restart API; use `frontend/src/services/webrtcClient.ts` (REST remains default transport in UI until wired end-to-end).
9. **Export transcript**: use the button on the detect screen (downloads JSON).

## Implemented vs deferred fallback matrix

| Layer | Implemented in this branch | Deferred / docs-only |
|------|----------------------------|----------------------|
| Streaming | REST `/predict` (always) | Full WebRTC ICE trickle + TURN hardening |
| TTS | Browser + `/tts` (optional gTTS / ElevenLabs) | Coqui local server bundle |
| LLM | `/llm/correct` OpenAI→Ollama | Rule-based grammar module |
| Translation | `/translate` proxy + optional `public/gloss/isl_gloss.json` | Google Cloud Translation proxy |
| Vision | `/fallback` OpenAI vision if key present | Gemini path |

## Security notes

- Never commit API keys. Use `.env` / CI secrets.
- Treat all client input as untrusted; backend routes validate lengths and use timeouts (`httpx`).
- Prefer `ALLOWED_ORIGINS` not `*` in production.

## Tests

Frontend:

```powershell
cd frontend
npm test
```

## Troubleshooting

- **WebRTC disabled log** (`aiortc not available`): install `requirements-webrtc.txt` into the same venv used to run the API.
- **`/tts` returns 501**: expected until `ENABLE_GTTS=1` or ElevenLabs env vars are configured; the UI falls back to browser TTS for `server`/`elevenlabs` provider paths.
- **TensorFlow logs on import**: benign; set `TF_CPP_MIN_LOG_LEVEL=3` to reduce noise.
