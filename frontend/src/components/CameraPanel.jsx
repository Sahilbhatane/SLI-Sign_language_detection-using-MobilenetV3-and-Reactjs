import React, { useCallback, useEffect, useRef, useState } from 'react';

import WebcamCapture from './WebcamCapture';
import { startWebRtcSession } from '../services/webrtcClient';
import { normalizeHandOverlayList } from '../utils/handOverlay';

/**
 * Orchestrates WebRTC vs REST: on detection start, try WebRTC; on failure or disconnect, REST polling resumes.
 */
const CameraPanel = ({
  onDetection,
  isActive,
  transport,
  onTransportChange,
  onFpsSample,
  onWebRtcFallback,
}) => {
  const streamRef = useRef(null);
  const sessionRef = useRef(null);
  const startGenRef = useRef(0);
  const fallbackReasonRef = useRef(null);
  const [useRestPolling, setUseRestPolling] = useState(true);
  const [handOverlay, setHandOverlay] = useState(null);

  const resetFallbackReason = useCallback(() => {
    fallbackReasonRef.current = null;
    onWebRtcFallback?.(null);
  }, [onWebRtcFallback]);

  const reportFallbackReason = useCallback(
    (reason) => {
      if (!reason) return;
      fallbackReasonRef.current = reason;
      onWebRtcFallback?.(reason);
    },
    [onWebRtcFallback]
  );

  const stopWebRtc = useCallback(() => {
    try {
      sessionRef.current?.close();
    } catch {
      /* ignore */
    }
    sessionRef.current = null;
    setUseRestPolling(true);
    onTransportChange?.('rest');
  }, [onTransportChange]);

  const onPred = useCallback(
    (msg) => {
      if (!msg?.success) return;
      const c = Number(msg.confidence) ?? 0;
      const confidence01 = c > 1 ? c / 100 : c;
      if (Array.isArray(msg.hands) && msg.hands.length > 0) {
        const hands = normalizeHandOverlayList(msg);
        setHandOverlay(hands.length > 0 ? { hands } : null);
      } else if (Array.isArray(msg.hand_bbox_norm) && msg.hand_bbox_norm.length === 4) {
        setHandOverlay({
          hands: [
            {
              bbox: msg.hand_bbox_norm,
              landmarks: Array.isArray(msg.hand_landmarks_norm) ? msg.hand_landmarks_norm : null,
            },
          ],
        });
      } else {
        setHandOverlay(null);
      }
      onDetection({
        phrase: msg.prediction,
        confidence: confidence01,
        allPredictions: Array.isArray(msg.predictions) ? msg.predictions : [],
        timestamp: new Date().toISOString(),
        processingTimeMs: undefined,
        minConfidence: undefined,
      });
    },
    [onDetection]
  );

  const handleCapturingChange = useCallback(
    async (capturing) => {
      if (!capturing) {
        stopWebRtc();
        return;
      }

      const gen = ++startGenRef.current;
      resetFallbackReason();
      const stream = streamRef.current;
      if (!stream) {
        setUseRestPolling(true);
        onTransportChange?.('rest');
        return;
      }

      try {
        const session = await startWebRtcSession(stream, onPred, {
          onFallback: (reason) => {
            reportFallbackReason(reason);
            sessionRef.current = null;
            setUseRestPolling(true);
            onTransportChange?.('rest');
          },
        });
        if (gen !== startGenRef.current) {
          session.close();
          return;
        }
        sessionRef.current = session;
        setUseRestPolling(false);
        onTransportChange?.('webrtc');
      } catch (e) {
        if (gen !== startGenRef.current) return;
        console.warn('WebRTC session failed, using REST polling', e);
        if (!fallbackReasonRef.current) {
          reportFallbackReason('webrtc:error');
        }
        sessionRef.current = null;
        setUseRestPolling(true);
        onTransportChange?.('rest');
      }
    },
    [onPred, onTransportChange, reportFallbackReason, resetFallbackReason, stopWebRtc]
  );

  const handleUserMedia = useCallback((stream) => {
    streamRef.current = stream;
  }, []);

  useEffect(() => () => stopWebRtc(), [stopWebRtc]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-gray-400 px-1">
        <span>Camera</span>
        <span className="rounded-full bg-gray-800 px-2 py-0.5 border border-gray-700">
          transport: <span className="text-gray-200 font-mono">{transport}</span>
        </span>
      </div>
      <WebcamCapture
        onDetection={onDetection}
        isActive={isActive}
        onUserMedia={handleUserMedia}
        useRestPolling={useRestPolling}
        onCapturingChange={handleCapturingChange}
        onFpsSample={onFpsSample}
        handOverlay={handOverlay}
      />
    </div>
  );
};

export default CameraPanel;
