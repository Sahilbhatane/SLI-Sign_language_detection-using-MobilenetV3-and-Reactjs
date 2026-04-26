/**
 * Minimal WebRTC client: connects to `/ws/webrtc`, sends an SDP offer, receives an SDP answer,
 * then receives `prediction` JSON messages from the server.
 *
 * ICE trickle is not implemented yet; use a reachable network path or add TURN for production.
 */

export type PredictionMessage = {
  type: 'prediction';
  success: boolean;
  prediction: string;
  confidence: number;
  predictions: unknown;
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

export async function startWebRtcSession(
  stream: MediaStream,
  onPrediction: (msg: PredictionMessage) => void
): Promise<RTCPeerConnection> {
  const ws = new WebSocket(buildWsUrl());
  const pc = new RTCPeerConnection({ iceServers: parseIceServers() });

  stream.getTracks().forEach((t) => pc.addTrack(t, stream));

  await new Promise<void>((resolve, reject) => {
    ws.addEventListener('open', () => resolve(), { once: true });
    ws.addEventListener('error', () => reject(new Error('websocket_failed')), { once: true });
  });

  ws.addEventListener('message', (ev) => {
    try {
      const msg = JSON.parse(String(ev.data));
      if (msg?.type === 'prediction') {
        onPrediction(msg as PredictionMessage);
      }
    } catch {
      /* ignore */
    }
  });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp, sdpType: offer.type }));

  await new Promise<void>((resolve, reject) => {
    const onMsg = (ev: MessageEvent) => {
      try {
        const msg = JSON.parse(String(ev.data));
        if (msg?.type === 'answer' && msg.sdp) {
          ws.removeEventListener('message', onMsg);
          void pc.setRemoteDescription({ type: msg.sdpType || 'answer', sdp: msg.sdp }).then(resolve, reject);
        }
      } catch {
        /* ignore */
      }
    };
    ws.addEventListener('message', onMsg);
    setTimeout(() => reject(new Error('webrtc_answer_timeout')), 15_000);
  });

  return pc;
}
