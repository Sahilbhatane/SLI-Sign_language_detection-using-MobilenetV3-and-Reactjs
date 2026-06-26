"""
Optional WebRTC (aiortc) signaling + video-frame inference.

If `aiortc` is not installed, registration is skipped (REST remains the supported path).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


from ensemble_inference import mirror_overlay_hands_norm_x


def _overlay_json_payload(model: Any) -> Tuple[bool, Optional[List[float]], Optional[List[List[float]]], List[Dict[str, Any]]]:
    """Plain floats/lists for JSON (avoids numpy scalar serialization issues)."""
    hands = getattr(model, "get_overlay_hands_json", None)
    hands_list: List[Dict[str, Any]] = hands() if callable(hands) else []
    if not hands_list:
        hands_raw = getattr(model, "last_overlay_hands_norm", None) or []
        for h in hands_raw:
            b = h.get("bbox_norm")
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            lms = h.get("landmarks_norm")
            entry: Dict[str, Any] = {
                "bbox_norm": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
            }
            if isinstance(lms, list) and lms:
                entry["landmarks_norm"] = [[float(p[0]), float(p[1])] for p in lms[:21]]
            hands_list.append(entry)

    bbox = getattr(model, "last_overlay_bbox_norm", None)
    lms = getattr(model, "last_overlay_landmarks_norm", None)
    if not hands_list and (not bbox or len(bbox) != 4):
        return False, None, None, []
    bbox_list: Optional[List[float]] = None
    lms_list: Optional[List[List[float]]] = None
    if bbox and len(bbox) == 4:
        bbox_list = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    if lms and len(lms) >= 1:
        lms_list = [[float(p[0]), float(p[1])] for p in lms[:21]]
    if not hands_list and bbox_list:
        hands_list = [{"bbox_norm": bbox_list, "landmarks_norm": lms_list}]
    return len(hands_list) > 0, bbox_list, lms_list, hands_list


def _mirror_overlay_norm_x(model: Any) -> None:
    """Map landmark/bbox from horizontally flipped image back to original camera x (0–1)."""
    hands = getattr(model, "last_overlay_hands_norm", None)
    if hands:
        model.last_overlay_hands_norm = mirror_overlay_hands_norm_x(hands)
        sync = getattr(model, "_sync_legacy_overlay_fields", None)
        if callable(sync):
            sync()
        return
    b = getattr(model, "last_overlay_bbox_norm", None)
    if b is not None and len(b) == 4:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        model.last_overlay_bbox_norm = (1.0 - x2, y1, 1.0 - x1, y2)
    lms = getattr(model, "last_overlay_landmarks_norm", None)
    if lms:
        model.last_overlay_landmarks_norm = [[1.0 - float(p[0]), float(p[1])] for p in lms]


def register_webrtc(app: FastAPI, get_model: Callable[[], Any]) -> None:
    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning("aiortc not available; WebRTC disabled: %s", e)
        return

    import cv2  # type: ignore
    from PIL import Image

    router = APIRouter()

    @router.websocket("/ws/webrtc")
    async def webrtc_ws(ws: WebSocket) -> None:
        await ws.accept()
        pc: Optional[RTCPeerConnection] = None
        track_task: Optional[asyncio.Task] = None
        async def safe_send(obj: Dict[str, Any]) -> None:
            try:
                await ws.send_text(json.dumps(obj))
            except Exception:
                return

        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "offer":
                    offer = RTCSessionDescription(sdp=msg["sdp"], type=msg.get("sdpType", "offer"))
                    pc = RTCPeerConnection()

                    @pc.on("track")
                    async def on_track(track):  # noqa: ANN001
                        nonlocal track_task
                        if track.kind != "video":
                            return

                        async def loop():
                            frame_idx = 0
                            # Lower stride = smoother overlay + better static_image_mode behavior; cost CPU.
                            sample_n = max(1, int(os.getenv("WEBRTC_FRAME_STRIDE", "2")))
                            while True:
                                try:
                                    frame = await track.recv()
                                except Exception:
                                    return
                                frame_idx += 1
                                if frame_idx % sample_n != 0:
                                    continue
                                try:
                                    try:
                                        rgb = frame.to_ndarray(format="rgb24")
                                    except Exception:
                                        bgr = frame.to_ndarray(format="bgr24")
                                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                                except Exception:
                                    continue
                                rgb = np.ascontiguousarray(rgb)
                                model = get_model()
                                if model is None:
                                    await safe_send({"type": "error", "message": "model_not_loaded"})
                                    return
                                try:
                                    pil = Image.fromarray(rgb)
                                    arr = model.preprocess_image(pil)
                                    if (
                                        not getattr(model, "last_overlay_hands_norm", None)
                                        and getattr(model, "last_overlay_bbox_norm", None) is None
                                        and os.getenv("WEBRTC_TRY_FLIPPED", "1").strip() != "0"
                                    ):
                                        rgb_f = cv2.flip(rgb, 1)
                                        arr2 = model.preprocess_image(Image.fromarray(rgb_f))
                                        if getattr(model, "last_overlay_hands_norm", None) or getattr(
                                            model, "last_overlay_bbox_norm", None
                                        ) is not None:
                                            _mirror_overlay_norm_x(model)
                                            arr = arr2
                                    preds = model.predict_top_k(arr, k=3)
                                    top = preds[0]
                                    min_conf = float(os.getenv("WEBRTC_MIN_CONFIDENCE", "0.6"))
                                    pred_label = "Detecting..."
                                    pred_conf_percent = 0.0
                                    if top["confidence"] >= min_conf:
                                        pred_label = top["class"]
                                        pred_conf_percent = top["confidence_percent"]
                                    overlay_ok, bbox_list, lms_list, hands_list = _overlay_json_payload(model)
                                    await safe_send(
                                        {
                                            "type": "prediction",
                                            "success": True,
                                            "prediction": pred_label,
                                            "confidence": pred_conf_percent,
                                            "predictions": preds,
                                            "hand_detected": overlay_ok,
                                            "hands": hands_list,
                                            "hand_bbox_norm": bbox_list,
                                            "hand_landmarks_norm": lms_list,
                                        }
                                    )
                                except Exception as ex:
                                    logger.warning("WebRTC frame inference failed: %s", ex, exc_info=True)
                                    await safe_send({"type": "error", "message": str(ex)})

                        track_task = asyncio.create_task(loop())

                    await pc.setRemoteDescription(offer)
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await safe_send(
                        {"type": "answer", "sdp": pc.localDescription.sdp, "sdpType": pc.localDescription.type}
                    )

        except WebSocketDisconnect:
            pass
        finally:
            if track_task:
                track_task.cancel()
            if pc is not None:
                await pc.close()

    app.include_router(router)
