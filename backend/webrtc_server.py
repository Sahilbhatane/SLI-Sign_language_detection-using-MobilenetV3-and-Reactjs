"""
Optional WebRTC (aiortc) signaling + video-frame inference.

If `aiortc` is not installed, registration is skipped (REST remains the supported path).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def register_webrtc(app: FastAPI, get_model: Callable[[], Any]) -> None:
    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription  # type: ignore
        from aiortc.contrib.media import MediaBlackhole  # type: ignore
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
        blackhole = MediaBlackhole()

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
                    pc.addTransceiver("video", direction="recvonly")

                    @pc.on("track")
                    async def on_track(track):  # noqa: ANN001
                        nonlocal track_task
                        if track.kind != "video":
                            return
                        blackhole.addTrack(track)

                        async def loop():
                            frame_idx = 0
                            sample_n = int(os.getenv("WEBRTC_FRAME_STRIDE", "3"))
                            while True:
                                try:
                                    frame = await track.recv()
                                except Exception:
                                    return
                                frame_idx += 1
                                if frame_idx % sample_n != 0:
                                    continue
                                try:
                                    img = frame.to_ndarray(format="bgr24")
                                except Exception:
                                    continue
                                model = get_model()
                                if model is None:
                                    await safe_send({"type": "error", "message": "model_not_loaded"})
                                    return
                                try:
                                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    pil = Image.fromarray(rgb)
                                    arr = model.preprocess_image(pil)
                                    preds = model.predict_top_k(arr, k=3)
                                    top = preds[0]
                                    min_conf = float(os.getenv("WEBRTC_MIN_CONFIDENCE", "0.6"))
                                    pred_label = "Detecting..."
                                    pred_conf_percent = 0.0
                                    if top["confidence"] >= min_conf:
                                        pred_label = top["class"]
                                        pred_conf_percent = top["confidence_percent"]
                                    await safe_send(
                                        {
                                            "type": "prediction",
                                            "success": True,
                                            "prediction": pred_label,
                                            "confidence": pred_conf_percent,
                                            "predictions": preds,
                                        }
                                    )
                                except Exception as ex:
                                    await safe_send({"type": "error", "message": str(ex)})

                        track_task = asyncio.create_task(loop())

                    await pc.setRemoteDescription(offer)
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await safe_send({"type": "answer", "sdp": pc.localDescription.sdp, "sdpType": pc.localDescription.type})

        except WebSocketDisconnect:
            pass
        finally:
            if track_task:
                track_task.cancel()
            if pc is not None:
                await pc.close()

    app.include_router(router)
