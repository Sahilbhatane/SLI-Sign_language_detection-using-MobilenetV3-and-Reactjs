"""
Additional FastAPI routes: TTS, LLM correction, low-confidence fallback.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _token_jaccard(a: str, b: str) -> float:
    ta = {w for w in re.split(r"\s+", a.lower().strip()) if w}
    tb = {w for w in re.split(r"\s+", b.lower().strip()) if w}
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter) / float(union) if union else 0.0


def _sentence_segments_count(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    parts = re.split(r"[.!?]+(?:\s+|$)", s)
    n = len([p for p in parts if p.strip()])
    return max(1, n)


def _sanitize_llm_correction(original: str, corrected: str, max_predict: int) -> str:
    """Reject hallucinated drift: low overlap with original, extra sentences, or excessive length."""
    o = (original or "").strip()
    c = (corrected or "").strip()
    if not c:
        return o
    cap = min(max(len(o) * 2 + 80, 120), 2000, max_predict * 8)
    if len(c) > cap:
        c = c[:cap].rsplit(" ", 1)[0].strip()
    if _sentence_segments_count(c) > _sentence_segments_count(o) + 1:
        return o
    if _token_jaccard(o, c) < 0.7:
        return o
    return c


def _parse_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


class TtsRequest(BaseModel):
    text: str = Field(..., max_length=2000)
    lang: str = Field(default="en-US", max_length=32)
    format: str = Field(default="mp3", max_length=8)
    provider: str = Field(default="auto", max_length=32)


class LLMCorrectRequest(BaseModel):
    text: str = Field(..., max_length=4000)
    max_tokens: int = Field(default=256, ge=16, le=1024)


class FallbackRequest(BaseModel):
    image_b64: Optional[str] = None
    recent_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    reason: Optional[str] = Field(default=None, max_length=200)


class TranslateRequest(BaseModel):
    q: str = Field(..., max_length=8000)
    source: str = Field(default="en", max_length=16)
    target: str = Field(..., max_length=16)
    format: str = Field(default="text", max_length=16)


def _libretranslate_base() -> str:
    return os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.com").rstrip("/")


def _build_upstream_detail(
    code: str,
    message: str,
    *,
    upstream_status: Optional[int] = None,
    retry_after: Optional[str] = None,
) -> Dict[str, Any]:
    detail = {"code": code, "message": message}
    if upstream_status is not None:
        detail["upstream_status"] = upstream_status
    if retry_after:
        detail["retry_after"] = retry_after
    return detail


async def _libretranslate_json(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: float = 20.0,
) -> Any:
    base = _libretranslate_base()
    url = f"{base}{path}"
    request_kwargs: Dict[str, Any] = {}
    if json_body is not None:
        request_kwargs["json"] = json_body
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.request(method, url, **request_kwargs)
    except httpx.TimeoutException:
        logger.warning("LibreTranslate timeout on %s", url)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_build_upstream_detail("upstream_timeout", "LibreTranslate timed out"),
        )
    except httpx.RequestError as exc:
        logger.warning("LibreTranslate request error on %s: %s", url, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_build_upstream_detail("upstream_network_error", "LibreTranslate request failed"),
        )

    if r.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        retry_after = r.headers.get("Retry-After")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=_build_upstream_detail(
                "upstream_rate_limited",
                "LibreTranslate rate limited",
                upstream_status=r.status_code,
                retry_after=retry_after,
            ),
            headers={"Retry-After": retry_after} if retry_after is not None else None,
        )
    if 400 <= r.status_code < 500:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_build_upstream_detail(
                "upstream_4xx",
                "LibreTranslate rejected the request",
                upstream_status=r.status_code,
            ),
        )
    if r.status_code >= 500:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_build_upstream_detail(
                "upstream_5xx",
                "LibreTranslate upstream error",
                upstream_status=r.status_code,
            ),
        )

    try:
        return r.json()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_build_upstream_detail("upstream_invalid_json", "LibreTranslate returned invalid JSON"),
        )


async def _openai_correct_text(text: str, max_tokens: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing_openai")

    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini").strip()
    url = "https://api.openai.com/v1/chat/completions"
    system = (
        "You are a careful editor. Fix grammar and fluency only. "
        "Do not change meaning, facts, or intent. If unsure, return the original text unchanged. "
        "Return ONLY the corrected text without quotes."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return str(data["choices"][0]["message"]["content"]).strip()


async def _ollama_correct_text(text: str, max_tokens: int) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1").strip()
    url = f"{base}/api/generate"
    prompt = (
        "Fix grammar only; do not change meaning or facts. "
        "Return ONLY the corrected text.\n\n"
        f"TEXT:\n{text}\n"
    )
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": max_tokens}}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return str(data.get("response", "")).strip()


async def _openai_vision_caption(image_b64: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing_openai")
    model = os.getenv("OPENAI_VLM_MODEL", "gpt-4o").strip()
    url = "https://api.openai.com/v1/chat/completions"
    system = (
        "You are assisting sign-language recognition. Describe ONLY what is visually present in the image "
        "that could help interpret a sign. If unclear, say 'unclear'. Do not invent glosses."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What sign or hands/gesture cues are visible?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
        "max_tokens": 200,
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return str(data["choices"][0]["message"]["content"]).strip()


def register_extra_routes(app: FastAPI) -> None:
    router = APIRouter()

    @router.post("/translate", tags=["Translation"])
    async def translate_proxy(req: TranslateRequest) -> Dict[str, Any]:
        return await _libretranslate_json(
            "POST",
            "/translate",
            json_body={"q": req.q, "source": req.source, "target": req.target, "format": req.format},
        )

    @router.get("/translate/languages", tags=["Translation"])
    async def translate_languages() -> Any:
        return await _libretranslate_json("GET", "/languages")

    @router.post("/tts", tags=["TTS"])
    async def tts_endpoint(req: TtsRequest) -> Response:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")

        provider = req.provider.strip().lower()
        if provider in ("auto", "elevenlabs"):
            key = os.getenv("ELEVENLABS_API_KEY", "").strip()
            voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
            if key and voice_id:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                headers = {"xi-api-key": key, "Accept": "audio/mpeg", "Content-Type": "application/json"}
                payload = {"text": text, "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")}
                async with httpx.AsyncClient(timeout=60.0) as client:
                    r = await client.post(url, headers=headers, json=payload)
                    if r.status_code == 200 and r.content:
                        return Response(content=r.content, media_type="audio/mpeg")
                    logger.warning("ElevenLabs TTS failed: %s %s", r.status_code, r.text[:200])

        if _parse_bool_env("ENABLE_GTTS", "0"):
            try:
                from gtts import gTTS  # type: ignore

                lang = req.lang.split("-")[0] if "-" in req.lang else req.lang
                mp3_io = __import__("io").BytesIO()
                gTTS(text=text, lang=lang or "en").write_to_fp(mp3_io)
                data = mp3_io.getvalue()
                if data:
                    return Response(content=data, media_type="audio/mpeg")
            except Exception as e:
                logger.warning("gTTS failed: %s", e)

        raise HTTPException(
            status_code=501,
            detail={"code": "tts_not_configured", "message": "Configure ENABLE_GTTS=1 or ElevenLabs env vars."},
        )

    @router.post("/llm/correct", tags=["LLM"])
    async def llm_correct(req: LLMCorrectRequest) -> Dict[str, Any]:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")

        provider_used = "none"
        skipped_reason: Optional[str] = None
        corrected = text
        predict_cap = min(req.max_tokens, max(len(text) // 5 + 96, 128))

        try:
            corrected = await _openai_correct_text(text, predict_cap)
            provider_used = "openai"
        except Exception as e_openai:
            skipped_reason = f"openai:{e_openai}"
            try:
                corrected = await _ollama_correct_text(text, predict_cap)
                provider_used = "ollama"
                skipped_reason = None
            except Exception as e_ollama:
                corrected = text
                provider_used = "none"
                skipped_reason = f"ollama:{e_ollama}"

        corrected = _sanitize_llm_correction(text, corrected, predict_cap)

        return {"corrected": corrected, "provider_used": provider_used, "skipped_reason": skipped_reason}

    @router.post("/fallback", tags=["Fallback"])
    async def fallback_endpoint(req: FallbackRequest) -> Dict[str, Any]:
        image = (req.image_b64 or "").strip()
        if image and "," in image:
            image = image.split(",", 1)[1]

        if image and os.getenv("OPENAI_API_KEY", "").strip():
            try:
                caption = await _openai_vision_caption(image)
                return {"used_fallback": True, "provider_used": "openai_vision", "summary": caption}
            except Exception as e:
                logger.info("vision fallback failed: %s", e)

        if req.recent_predictions:
            try:
                corrected = await _ollama_correct_text(
                    "Given weak sign predictions as JSON, suggest the most likely short gloss label only, no prose:\n"
                    + json.dumps(req.recent_predictions)[:2000],
                    128,
                )
                return {"used_fallback": True, "provider_used": "ollama", "summary": corrected}
            except Exception:
                pass

        return {
            "used_fallback": False,
            "provider_used": "none",
            "summary": "",
            "disabled_reason": "no_vision_or_llm_configured",
        }

    app.include_router(router)
