/**
 * WebRTC client: `/ws/webrtc` signaling, then `prediction` JSON over WebSocket.
 * ICE trickle is not implemented; use TURN for strict NAT environments.
 */

export type PredictionMessage = {
  type: 'prediction';
  success: boolean;
  prediction: string;
  confidence: number;
  predictions: unknown;
  /** True when MediaPipe found a hand for this frame (same crop as ONNX). */
  hand_detected?: boolean;
  /** [x1, y1, x2, y2] in 0–1 relative to full frame (before crop). */
  hand_bbox_norm?: number[] | null;
  /** 21 points [[x,y], ...] in 0–1 image space. */
  hand_landmarks_norm?: [number, number][] | null;
};

export type WebRtcSession = {
  pc: RTCPeerConnection;
  close: () => void;
};

export type StartWebRtcOptions = {
  /** Called when ICE/PC fails, WS errors, or connection drops (idempotent). */
  onFallback?: (reason: string) => void;
};

function buildWsUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws/webrtc`;
}

function parseIceServers(): RTCIceServer[] {
  const raw = import.meta.env.VITE_STUN_URLS as string | undefined;
  if (!raw || !raw.trim()) {
    return [{ urls: 'stun:stun.l.google.com:19302' }];
  }
  return raw.split(',').map((u) => ({ urls: u.trim() }));
}

/**
 * Starts a WebRTC session: adds tracks from `stream`, negotiates over WebSocket, then delivers predictions.
 * Call `close()` to release PC/WS and stop inference callbacks.
 */
export async function startWebRtcSession(
  stream: MediaStream,
  onPrediction: (msg: PredictionMessage) => void,
  options?: StartWebRtcOptions
): Promise<WebRtcSession> {
  const onFallback = options?.onFallback;
  const ws = new WebSocket(buildWsUrl());
  const pc = new RTCPeerConnection({ iceServers: parseIceServers() });

  let closed = false;
  let fallbackNotified = false;

  const notifyFallback = (reason: string) => {
    if (fallbackNotified) return;
    fallbackNotified = true;
    try {
      onFallback?.(reason);
    } catch {
      /* ignore */
    }
  };

  const onPredictionMsg = (ev: MessageEvent) => {
    try {
      const msg = JSON.parse(String(ev.data));
      if (msg?.type === 'prediction') {
        onPrediction(msg as PredictionMessage);
      }
    } catch {
      /* ignore */
    }
  };

  const close = () => {
    if (closed) return;
    closed = true;
    try {
      ws.removeEventListener('message', onPredictionMsg);
    } catch {
      /* ignore */
    }
    try {
      ws.close();
    } catch {
      /* ignore */
    }
    try {
      pc.getSenders().forEach((s) => {
        try {
          pc.removeTrack(s);
        } catch {
          /* ignore */
        }
      });
    } catch {
      /* ignore */
    }
    try {
      pc.close();
    } catch {
      /* ignore */
    }
  };

  const safeFallback = (reason: string) => {
    notifyFallback(reason);
    close();
  };

  stream.getTracks().forEach((t) => pc.addTrack(t, stream));

  pc.addEventListener('connectionstatechange', () => {
    const st = pc.connectionState;
    if (st === 'failed' || st === 'disconnected') {
      safeFallback(`pc:${st}`);
    }
  });

  pc.addEventListener('iceconnectionstatechange', () => {
    const st = pc.iceConnectionState;
    if (st === 'failed' || st === 'disconnected') {
      safeFallback(`ice:${st}`);
    }
  });

  await new Promise<void>((resolve, reject) => {
    ws.addEventListener('open', () => resolve(), { once: true });
    ws.addEventListener('error', () => reject(new Error('websocket_failed')), { once: true });
  });

  ws.addEventListener('close', () => {
    if (!closed) safeFallback('ws:close');
  });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp, sdpType: offer.type }));

  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const onMsg = (ev: MessageEvent) => {
      try {
        const msg = JSON.parse(String(ev.data));
        if (msg?.type === 'answer' && msg.sdp) {
          ws.removeEventListener('message', onMsg);
          window.clearTimeout(timeoutId);
          void pc
            .setRemoteDescription({ type: msg.sdpType || 'answer', sdp: msg.sdp })
            .then(() => {
              if (settled) return;
              settled = true;
              ws.addEventListener('message', onPredictionMsg);
              resolve();
            })
            .catch((e) => {
              if (settled) return;
              settled = true;
              safeFallback('sdp_set_failed');
              reject(e instanceof Error ? e : new Error('sdp_set_failed'));
            });
        }
      } catch {
        /* ignore */
      }
    };
    ws.addEventListener('message', onMsg);
    const timeoutId = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      ws.removeEventListener('message', onMsg);
      safeFallback('answer_timeout');
      reject(new Error('webrtc_answer_timeout'));
    }, 15_000);
  });

  return { pc, close };
}
