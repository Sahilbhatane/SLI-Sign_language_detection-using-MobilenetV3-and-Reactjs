import React, { useRef, useState, useCallback, useEffect, useLayoutEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { MP_HAND_CONNECTIONS } from '../utils/mpHandConnections';

/**
 * Map normalized image coords (0–1) to canvas pixels for CSS object-fit: cover.
 * Uses intrinsic size when available; falls back to element size so drawing works before metadata loads.
 */
function mapNormToCanvasIntrinsic(nx, ny, vw, vh, cw, ch) {
  if (!vw || !vh || !cw || !ch) return [0, 0];
  const scale = Math.max(cw / vw, ch / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;
  return [nx * vw * scale + ox, ny * vh * scale + oy];
}

const WebcamCapture = ({
  onDetection,
  isActive,
  onUserMedia,
  useRestPolling = true,
  onCapturingChange,
  onFpsSample,
  /** @type {{ bbox: number[], landmarks: [number, number][] | null } | null} */
  handOverlay = null,
}) => {
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6);
  const webcamRef = useRef(null);
  const overlayWrapRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState(null);
  const [lastCapture, setLastCapture] = useState(null);
  const [capturesPerSec, setCapturesPerSec] = useState(0);
  const intervalRef = useRef(null);
  const fpsTickRef = useRef(0);
  const fpsIntervalRef = useRef(null);
  const captureAndPredict = useCallback(async () => {
    if (typeof document !== 'undefined' && document.hidden) {
      return;
    }
    if (!webcamRef.current || !isActive) {
      return;
    }

    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        return;
      }

      setLastCapture(Date.now());
      fpsTickRef.current += 1;

      const base64Image = imageSrc.replace(/^data:image\/\w+;base64,/, '');

      const response = await axios.post(
        '/api/predict',
        {
          image: base64Image,
          top_k: 3,
          min_confidence: confidenceThreshold,
        },
        { timeout: 5000 }
      );

      if (response.data && response.data.predictions) {
        const pred = response.data.prediction;
        const confPercent = Number(response.data.confidence) || 0;
        const confidence01 = confPercent / 100;

        onDetection({
          phrase: pred,
          confidence: confidence01,
          allPredictions: response.data.predictions,
          timestamp: new Date().toISOString(),
          processingTimeMs: response.data.processing_time_ms,
          minConfidence: response.data.min_confidence ?? confidenceThreshold,
        });
        setError(null);
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to detect sign');
    }
  }, [isActive, onDetection, confidenceThreshold]);

  useEffect(() => {
    onCapturingChange?.(capturing);
  }, [capturing, onCapturingChange]);

  useEffect(() => {
    if (!isActive || !capturing || !useRestPolling) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (fpsIntervalRef.current) clearInterval(fpsIntervalRef.current);
      return undefined;
    }

    intervalRef.current = setInterval(() => {
      captureAndPredict();
    }, 250);

    fpsIntervalRef.current = setInterval(() => {
      const n = fpsTickRef.current;
      setCapturesPerSec(n);
      onFpsSample?.(n);
      fpsTickRef.current = 0;
    }, 1000);

    return () => {
      clearInterval(intervalRef.current);
      clearInterval(fpsIntervalRef.current);
    };
  }, [isActive, capturing, useRestPolling, captureAndPredict, onFpsSample]);

  useLayoutEffect(() => {
    const canvas = overlayCanvasRef.current;
    const wrap = overlayWrapRef.current;
    const video = webcamRef.current?.video;
    if (!canvas || !wrap || !video) return undefined;

    const dpr = typeof globalThis.window !== 'undefined' ? globalThis.window.devicePixelRatio || 1 : 1;
    let rafId = 0;

    const draw = () => {
      const { clientWidth: cw, clientHeight: ch } = wrap;
      if (!cw || !ch) return;
      canvas.width = Math.floor(cw * dpr);
      canvas.height = Math.floor(ch * dpr);
      canvas.style.width = `${cw}px`;
      canvas.style.height = `${ch}px`;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cw, ch);

      if (!capturing) return;

      const vw = video.videoWidth || video.clientWidth || 640;
      const vh = video.videoHeight || video.clientHeight || 480;

      const bbox = handOverlay?.bbox;
      const landmarks = handOverlay?.landmarks;

      if (bbox && bbox.length === 4) {
        const [x1, y1, x2, y2] = bbox;
        const [px1, py1] = mapNormToCanvasIntrinsic(x1, y1, vw, vh, cw, ch);
        const [px2, py2] = mapNormToCanvasIntrinsic(x2, y2, vw, vh, cw, ch);
        const left = Math.min(px1, px2);
        const top = Math.min(py1, py2);
        const w = Math.abs(px2 - px1);
        const h = Math.abs(py2 - py1);

        ctx.strokeStyle = 'rgba(34, 211, 238, 0.95)';
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        ctx.strokeRect(left, top, w, h);

        if (Array.isArray(landmarks) && landmarks.length >= 21) {
          ctx.strokeStyle = 'rgba(52, 211, 153, 0.85)';
          ctx.lineWidth = 2;
          for (const [a, b] of MP_HAND_CONNECTIONS) {
            const pa = landmarks[a];
            const pb = landmarks[b];
            if (!pa || !pb) continue;
            const [ax, ay] = mapNormToCanvasIntrinsic(pa[0], pa[1], vw, vh, cw, ch);
            const [bx, by] = mapNormToCanvasIntrinsic(pb[0], pb[1], vw, vh, cw, ch);
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.stroke();
          }
          ctx.fillStyle = 'rgba(250, 250, 250, 0.95)';
          for (const p of landmarks) {
            if (!p) continue;
            const [px, py] = mapNormToCanvasIntrinsic(p[0], p[1], vw, vh, cw, ch);
            ctx.beginPath();
            ctx.arc(px, py, 3.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      } else if (!useRestPolling) {
        ctx.font = '600 13px system-ui, sans-serif';
        ctx.fillStyle = 'rgba(250, 204, 21, 0.95)';
        ctx.textAlign = 'center';
        ctx.fillText('No hand detected — show your hand to the camera', cw / 2, ch - 16);
      }
    };

    draw();
    const ro = new ResizeObserver(() => draw());
    ro.observe(wrap);
    const onVideo = () => draw();
    video.addEventListener('loadedmetadata', onVideo);

    const runRaf = !useRestPolling && capturing;
    let rafStopped = false;
    const tick = () => {
      if (rafStopped) return;
      draw();
      rafId = globalThis.requestAnimationFrame(tick);
    };
    if (runRaf) {
      rafId = globalThis.requestAnimationFrame(tick);
    }

    return () => {
      rafStopped = true;
      globalThis.cancelAnimationFrame(rafId);
      ro.disconnect();
      video.removeEventListener('loadedmetadata', onVideo);
    };
  }, [capturing, handOverlay, useRestPolling]);

  const toggleCapture = () => {
    setCapturing(!capturing);
    if (!capturing) {
      setError(null);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="relative bg-gray-900 rounded-2xl overflow-hidden shadow-2xl transition-opacity duration-200">
        <div ref={overlayWrapRef} className="relative aspect-video bg-black">
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={{
              width: 640,
              height: 480,
              facingMode: 'user',
            }}
            mirrored={false}
            className="w-full h-full object-cover"
            style={{ transform: 'scaleX(1)' }}
            onUserMedia={(stream) => {
              onUserMedia?.(stream);
            }}
            onUserMediaError={() => {
              setError('Failed to access webcam. Please check permissions.');
            }}
          />

          {capturing && (
            <div className="absolute inset-0 pointer-events-none transition-opacity duration-200">
              <div className="absolute top-4 left-4 z-10 flex items-center space-x-2 bg-red-600 text-white px-3 py-2 rounded-full">
                <span className="w-3 h-3 bg-white rounded-full animate-pulse" />
                <span className="text-sm font-semibold">DETECTING</span>
              </div>
              <canvas
                ref={overlayCanvasRef}
                className="absolute inset-0 z-[5] w-full h-full pointer-events-none"
                role="presentation"
              />
            </div>
          )}

          {error && (
            <div className="absolute top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg max-w-sm transition-opacity duration-200">
              <p className="text-sm">{error}</p>
            </div>
          )}
        </div>

        <div className="bg-gray-800 px-6 py-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-4">
            <button
              type="button"
              onClick={toggleCapture}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-200 ${
                capturing
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {capturing ? '⏸ Stop Detection' : '▶ Start Detection'}
            </button>

            <button
              type="button"
              onClick={captureAndPredict}
              disabled={!isActive}
              className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-xl font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              📸 Capture Frame
            </button>

            <div className="flex items-center space-x-2">
              <label htmlFor="confidence-threshold" className="text-gray-300 text-sm whitespace-nowrap">
                Min confidence: {(confidenceThreshold * 100).toFixed(0)}%
              </label>
              <input
                id="confidence-threshold"
                type="range"
                min="0"
                max="100"
                step="5"
                value={Math.round(confidenceThreshold * 100)}
                onChange={(e) => setConfidenceThreshold(Number(e.target.value) / 100)}
                className="w-32 accent-blue-500"
                title="Sent to API as min_confidence"
              />
            </div>
          </div>

          <div className="flex items-center space-x-6 text-sm text-gray-300">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-400' : 'bg-gray-500'}`} />
              <span>{isActive ? 'Backend Connected' : 'Backend Offline'}</span>
            </div>
            {capturing && (
              <span className="text-gray-400">~{capturesPerSec}/s captures</span>
            )}
            {lastCapture && (
              <span>Last: {new Date(lastCapture).toLocaleTimeString()}</span>
            )}
          </div>
        </div>
      </div>

      <div className="mt-4 bg-gray-800 rounded-xl p-4 transition-opacity duration-200">
        <h3 className="text-white font-semibold mb-2 flex items-center">
          <span className="text-xl mr-2">💡</span>
          Tips for Best Results
        </h3>
        <ul className="text-gray-300 text-sm space-y-1">
          <li>• Ensure good lighting on your hands</li>
          <li>• In WebRTC mode, follow the cyan box and green hand skeleton on your hand</li>
          <li>• Hold each sign steady for 2–3 seconds</li>
          <li>• Position yourself centered in the webcam</li>
        </ul>
      </div>
    </div>
  );
};

export default WebcamCapture;
