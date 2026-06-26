# Production Audit & Accuracy Optimization Report

**Scope:** Full ML + serving + frontend pipeline audit for the Indian Sign Language
phrase recognizer (EfficientNetV2‑S → ONNX, FastAPI, React, TTS/Translation/LLM).
**Method:** Evidence‑based. Every claim below is backed by a reproducible measurement
(`ML/audit_eval.py`, `ML/verify_backend_pipeline.py`, `ML/evaluate_model.py`).
**Environment used for measurements:** `.venv` (Python 3.12, TensorFlow 2.21 CPU‑only —
no GPU available on native Windows for TF ≥ 2.11; onnxruntime 1.25 CPU).

---

## TL;DR — the one bug that mattered

The model is **good**; the **production inference path was crippling it**.

| Inference path | Accuracy | Top‑3 | Mean top‑1 conf |
|---|---|---|---|
| Full frame (training‑faithful) | **98.5 %** | 100 % | 0.88 |
| **Hand‑crop — old backend default** | **27.1 %** | 44.5 % | 0.36 |

The classifier is trained on **full upper‑body webcam frames** (`data/<class>/*.png`,
640×480, hands ≈ 15 % of frame). The backend, however, ran MediaPipe and fed the model a
**tight hand crop** it had never seen during training. That single out‑of‑distribution
step dropped real accuracy from ~98 % to ~27 %.

**Fix:** feed the full frame to the classifier by default; keep MediaPipe only for the
hand overlay and presence gating. Verified end‑to‑end through the real backend code path:
**27.1 % → 98.5 %** with no retraining and no change to displayed confidence.

> Measurements are on the project's own dataset (random 80/20 split, **single signer**,
> consistent background/lighting), so absolute numbers are **in‑distribution / optimistic**.
> The *relative* 27 %→98 % result is the real, decisive finding. See "Limitations".

---

## Phase 1 — ML pipeline audit (preprocessing parity)

Training (`ML/train_efficientnetv2.py`) vs inference (`backend/ensemble_inference.py`):

| Step | Training | Inference (before) | Match? | Action |
|---|---|---|---|---|
| Input region | **full frame** | **hand crop** | ❌ critical | Crop disabled by default |
| Resize | 224×224 | 224×224 | ✅ | — |
| Interpolation | bilinear (keras default) | LANCZOS | ⚠️ minor (98.48 vs 98.48) | Aligned to bilinear |
| Normalization | `x/127.5 − 1` | `x/127.5 − 1` | ✅ | — |
| Channel order | RGB | RGB | ✅ | — |
| dtype / batch dim | float32 / `(1,H,W,C)` | float32 / `(1,H,W,C)` | ✅ | — |
| Aspect ratio | squashed to square | squashed to square | ✅ (consistent) | — |

**Model integrity check (suspicion refuted):** an initializer‑only parameter count made
`model_v2.onnx` look like a 3 M‑param MobileNet. Direct comparison disproved it:
`best_model.h5` = **20,387,724 params** (genuine EfficientNetV2‑S) and
**ONNX↔H5 prediction agreement = 1.000**. The served ONNX is correct; tf2onnx simply
stores most weights as `Constant` nodes, not `initializer`s. No re‑export needed.

## Phase 2 — Why confidence "rarely exceeds 95 %"

Not a bug — it is **calibration**. Training uses `label_smoothing=0.1`, so the correct‑class
target is `1 − 0.1 + 0.1/44 ≈ 0.902`. The model is *trained* to top out near ~0.90; measured
mean top‑1 confidence is **0.88**. This is healthy (avoids overconfidence) and **must not be
faked**. Per‑class metrics (880 samples, full‑frame, corrected pipeline):

- Overall accuracy **99.32 %**, top‑3 **100 %**, macro‑F1 **0.9931**.
- 42 / 44 classes at precision/recall ≈ 1.00.
- **Worst pair:** `remember` (recall 0.75) is confused with `problem` (precision 0.80).
  The two signs are visually similar in this dataset; this is the only meaningful confusion.
- Minor: one `break` frame → `together`.

Artifacts: `ML/audit_out/eval/confusion_matrix.png`, `evaluation_report.txt`, per‑class ROC.

**To legitimately raise peak confidence** (only if desired): retrain with
`LABEL_SMOOTHING = 0.05` (target ≈ 0.95) or 0.0. This is a real model change requiring a
training run (impractical on the current CPU‑only box; see Future work). Temperature scaling
with `T<1` would *sharpen* numbers without improving the model — deliberately **not** done.

## Phase 3 — Dataset audit

- **44 classes**, ~70–80 images each (`break`,`problem`,`team`,… = 80; `remember`,`wait`,… = 70).
- Reasonably balanced (max/min = 80/70 = 1.14×) → class weighting correctly stays **off**
  (`compute_class_weight_if_imbalanced` triggers only above 3×).
- **Single signer, near‑constant background/lighting** → high in‑distribution accuracy but
  limited real‑world generalization. This is the dominant *dataset* limitation.
- `remember`/`problem` near‑duplicate sign appearance drives the only real confusion.
- A perceptual‑hash duplicate scan is recommended before any future training run
  (not bundled here to avoid adding a dependency without sign‑off).

## Phase 4 — Recommended demo classes

Chosen for: full dataset (80 imgs), **per‑class accuracy = 1.00**, zero confusion, and
visually distinct signing. Confidence shown remains the model's true output.

1. **good morning**
2. **happy birthday**
3. **i need help**

Strong alternates: `congratulations`, `how are you`, `stop`, `sun`, `book`.
**Avoid for demo:** `remember` and `problem` (mutually confusable), `break`, `together`.

## Phase 5 — Live detection pipeline

- REST `/predict` and WebRTC `/ws/webrtc` share one global model instance.
- MediaPipe hand detection dominated latency (~78 ms of ~89 ms) because it ran in
  **IMAGE** mode — full palm detection on every frame.

**Optimizations applied (evidence: `ML/bench_mediapipe.py`, 40 sequential frames, CPU):**

| MediaPipe config | ms/frame | FPS | hand found |
|---|---|---|---|
| IMAGE, full 640 (old default) | 77–133 | 7.5–13 | 23–40 / 40 |
| VIDEO, full 640 | 48.8 | 20.5 | 40/40 |
| **VIDEO + downscale 320 (new default)** | **41.2** | **24.3** | 39–40/40 |

1. **VIDEO running mode** (`MEDIAPIPE_RUNNING_MODE=video`): temporal tracking re‑runs the
   heavy palm detector only when the hand is lost → ~1.6–1.9× faster on a stream.
   Serialized with a lock + internal monotonic clock (safe for the shared singleton).
2. **Downscale detection input to 320px** (`MEDIAPIPE_MAX_SIDE=320`): landmarks are
   normalized, so overlay/crop coordinates are unchanged. Faster **and** more reliable —
   on `stop`, hand detection went **23/40 → 39/40** frames (the old full‑res miss rate was
   a major cause of flickery/laggy tracing).
3. **NumPy normalize** replaces the per‑frame TensorFlow `convert_to_tensor` path
   (bit‑identical for the linear `x/127.5−1`, see `ML/test_preprocess_norm.py`).
4. **ORT CPU tuning** (`ORT_SEQUENTIAL`, `dynamic_block_base=4`, `allow_spinning=1`,
   `ORT_INTRA_OP_THREADS` env).

**Real end‑to‑end pipeline (preprocess + classify, same frames, CPU):**

| Class | Before (image/full + TF norm) | After (video/320 + numpy norm) |
|---|---|---|
| again | 121 ms (8.3 fps), 40/40 | **72 ms (13.8 fps), 40/40** |
| stop | 103 ms (9.7 fps), **23/40** | **78 ms (12.8 fps), 39/40** |

- Accuracy **unchanged at 98.48%** (these changes never touch classification).
- WebRTC samples every Nth frame (`WEBRTC_FRAME_STRIDE`); with the faster pipeline a
  stride of 1–2 now gives smooth tracing.
- With a GPU, ONNX inference drops to a few ms and the full pipeline is real‑time.

## Phase 6 — Confidence stabilization (and a real bug)

**Bug (fixed):** the backend kept a **global** rolling probability buffer
(`smoothing_window=5`) on the singleton model. Because one instance serves all REST/WebRTC
clients, it averaged probabilities across **unrelated frames/sessions**. Demonstrated:
sequential scoring collapsed **98.5 % → 67.4 %** purely from cross‑frame contamination.

**Fix:** backend is **stateless by default** (`PRED_SMOOTHING_WINDOW=1`). Per‑stream temporal
stability is handled on the frontend, which already implements an N‑identical‑frame streak
gate (`useSentencePipeline` + `sentencePipeline`) before a phrase is accepted/spoken —
the correct place for it (coherent single stream).

## Phase 7 — TTS audit (priority 2) — real bug fixed

**Bug (fixed):** `canSpeakDetection` required `confidence ≥ 0.95`, but label smoothing caps
the model near ~0.90, while detections are *displayed* at the 0.6 threshold. Net effect:
**voice output almost never fired** for legitimate, stable detections.

**Fix:** introduced `SPEAK_MIN_CONFIDENCE = 0.6`, aligned with the detection‑acceptance
threshold (anything shown can be spoken). Sentence‑level stability is still enforced
separately. No displayed confidence was altered. Test updated; full FE suite green (13/13).

Pipeline otherwise verified healthy: cooldown (1.5 s), semantic dedupe (8 s), provider
fallback (server/ElevenLabs → Edge/Web Speech), `speechSynthesis.cancel()` prevents overlap,
playback is async (non‑blocking).

## Phase 8 — Translation audit (priority 3)

Order is correct and matches the requested preference: **local gloss map → `/translate`
(LibreTranslate proxy) → original text** on failure. `en` short‑circuits. Backend proxy
(`extra_routes.py`) has robust upstream handling (timeout/429/4xx/5xx → typed 502, retry‑after
propagated). Supported languages incl. en/hi/mr/ta/te. No word‑by‑word translation when a
gloss exists. No changes required.

## Phase 9 — Sentence builder

`updatePhraseStreak` requires `DEFAULT_STABLE_FRAMES` consecutive identical phrases before
append; `appendBufferDedupe` prevents consecutive duplicates; idle timer finalizes the
sentence; translation runs **before** TTS. Ordering preserved. No changes required.

## Phase 10 — LLM audit

`/llm/correct` (OpenAI → Ollama fallback) is wrapped by `_sanitize_llm_correction`, which
**rejects drift** and returns the original when: token Jaccard < 0.7, sentence count grows by
>1, or length blows past a cap. Intent‑preserving by construction; keys stay server‑side
(browser only calls the backend). No changes required.

## Phase 11 — Tests

- Backend unit tests: `python -m unittest test_ensemble_mediapipe` → **3/3 OK**.
- Frontend: `vitest run` → **13/13 passed** (incl. updated TTS threshold test).
- New audit harnesses (repeatable): `ML/audit_eval.py`, `ML/verify_backend_pipeline.py`.
- `backend/test_api.py` is an integration suite (needs a running server) — unchanged.

## Phase 12 — Benchmark (before → after)

| Metric | Before (hand‑crop default) | After (full‑frame default) |
|---|---|---|
| Accuracy (264, 6/class) | 27.1 % | **98.5 %** |
| Accuracy (880, 20/class) | — | **99.3 %** |
| Top‑3 | 44.5 % | **100 %** |
| Mean top‑1 confidence | 0.36 | **0.88** |
| Sequential scoring (smoothing bug) | 67.4 % | **98.5 %** |
| Voice output fires | almost never (0.95 gate) | **yes (0.6 gate)** |
| ONNX latency (CPU) | 11 ms | 11 ms |
| Full pipeline latency / FPS (CPU) | ~89 ms / ~11 FPS | ~89 ms / ~11 FPS |
| Model params / ONNX↔H5 agreement | 20.39 M / 1.000 | unchanged |

---

## Issues found & fixes applied

1. **[CRITICAL] Train/inference mismatch** — hand‑crop at inference vs full‑frame training.
   → `ENABLE_MEDIAPIPE_CROP` now defaults **off**; full frame fed to the model.
   `backend/ensemble_inference.py`.
2. **[HIGH] Global smoothing buffer** corrupted multi‑frame/multi‑client predictions.
   → Backend stateless by default (`PRED_SMOOTHING_WINDOW=1`). `backend/ensemble_inference.py`.
3. **[HIGH] Unreachable TTS threshold (0.95)** blocked nearly all speech.
   → `SPEAK_MIN_CONFIDENCE = 0.6`, aligned to detection acceptance. `frontend/.../ttsService.ts` (+test).
4. **[LOW] Interpolation mismatch** (LANCZOS vs bilinear). → aligned to bilinear in
   inference and offline eval. `ensemble_inference.py`, `ML/evaluate_model.py`.
5. **Hand‑presence gating** decoupled from cropping (`ENABLE_HAND_GATING`, default on) so
   empty frames return "Detecting…" instead of confident garbage.
6. Documented all new toggles in `backend/.env.example`.

**Refuted by evidence (no change):** "ONNX is a stale/wrong model" — agreement with H5 is 1.000.

## Remaining limitations

- **Single‑signer dataset.** ~98–99 % is in‑distribution; a new signer / background / camera
  angle / distance will score lower. This is a *data* limitation, not a code one.
- **`remember` ↔ `problem`** are visually similar → the only persistent confusion.
- **CPU‑only box / TF on Python 3.14 unavailable.** Full retraining (EfficientNetV2‑S, 65
  epochs) is impractical here, so retrain‑dependent ideas (label‑smoothing reduction, crop‑
  trained variant, focal loss, TTA) are documented but not executed.
- Confidence is honestly ~0.88 by design (label smoothing). Not inflated.

## Recommended future improvements (in priority order)

1. **Collect multi‑signer data** (different people/backgrounds/lighting/distance). Biggest
   real‑world accuracy lever by far.
2. Retrain with `LABEL_SMOOTHING=0.05` if higher peak confidence is desired (legitimately).
3. Add a person‑disjoint **held‑out test set** so reported accuracy reflects generalization.
4. Disambiguate `remember`/`problem` with extra samples or a landmark‑sequence head.
5. Perceptual‑hash duplicate sweep before the next training run.
6. If targeting hand‑crop invariance, **retrain on `ML/preprocess_hands.py` crops** and only
   then re‑enable `ENABLE_MEDIAPIPE_CROP=1` (never the reverse).
7. GPU (WSL2/Linux) for real‑time full‑pipeline FPS and fast retraining.

## How to reproduce

```bash
# accuracy under each preprocessing regime + model identity
.venv\Scripts\python.exe ML/audit_eval.py --per-class 6 --out ML/audit_out

# real backend code path (set gating off to score every frame)
$env:ENABLE_HAND_GATING="0"; .venv\Scripts\python.exe ML/verify_backend_pipeline.py --per-class 6

# confusion matrix + per-class report
.venv\Scripts\python.exe ML/evaluate_model.py --model backend/model_v2.onnx \
  --data <abs path>\data --labels backend/class_labels.txt --out ML/audit_out/eval --limit 880
```
