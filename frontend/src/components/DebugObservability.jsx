import React from 'react';

/**
 * Compact dev-only overlay for transport, FPS, confidence, TTS, and LLM pipeline hints.
 */
export default function DebugObservability({
  transport,
  approxFps,
  lastConfidence,
  lastWebRtcFallback,
  ttsProvider,
  llmGrammarEnabled,
  llmFollowUpSpoke,
}) {
  if (!import.meta.env.DEV) return null;

  return (
    <div className="fixed bottom-3 right-3 z-50 max-w-xs rounded-lg border border-gray-600 bg-gray-950/95 px-3 py-2 text-[11px] font-mono text-gray-200 shadow-xl backdrop-blur-sm">
      <div className="text-amber-400/90 mb-1 font-semibold uppercase tracking-wide">Debug</div>
      <div>transport: {transport}</div>
      <div>webrtc fallback: {lastWebRtcFallback || '—'}</div>
      <div>fps ~: {typeof approxFps === 'number' ? approxFps : '—'}</div>
      <div>last conf: {lastConfidence != null ? Number(lastConfidence).toFixed(3) : '—'}</div>
      <div>tts: {ttsProvider}</div>
      <div>
        llm: {llmGrammarEnabled ? 'on' : 'off'}
        {llmGrammarEnabled ? ` / follow-up: ${llmFollowUpSpoke ? 'yes' : 'no'}` : ''}
      </div>
    </div>
  );
}
