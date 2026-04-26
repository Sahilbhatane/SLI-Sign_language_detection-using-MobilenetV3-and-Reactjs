import axios from 'axios';

import { DETECTING_PHRASE } from '../utils/signSpeech';

export type TtsProvider = 'edge' | 'server' | 'elevenlabs';

export type SpeakOptions = {
  /** When set, speech is allowed only if confidence >= 0.95 (model phrase path). */
  confidence?: number;
};

const DEFAULT_COOLDOWN_MS = 1500;

export function canSpeakDetection(input: { text: string; confidence: number }): boolean {
  const t = String(input.text || '').trim();
  if (!t || t === DETECTING_PHRASE) return false;
  return input.confidence >= 0.95;
}

export type TtsServiceDeps = {
  speakEdge?: (text: string, lang: string) => Promise<void>;
  speakServer?: (text: string, lang: string) => Promise<void>;
  cooldownMs?: number;
  now?: () => number;
};

async function defaultSpeakEdge(text: string, lang: string): Promise<void> {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang || 'en-US';
  u.rate = 1.0;
  u.pitch = 1.0;
  window.speechSynthesis.speak(u);
}

async function defaultSpeakServer(text: string, lang: string): Promise<void> {
  if (typeof window === 'undefined') return;
  const response = await axios.post(
    '/api/tts',
    { text, lang, format: 'mp3' },
    { responseType: 'blob', timeout: 60_000 }
  );
  const url = URL.createObjectURL(response.data);
  try {
    const audio = new Audio(url);
    await new Promise<void>((resolve, reject) => {
      audio.addEventListener('ended', () => resolve(), { once: true });
      audio.addEventListener('error', () => reject(new Error('audio playback failed')), { once: true });
      void audio.play().catch(reject);
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function createTtsService(deps: TtsServiceDeps = {}) {
  const cooldownMs = deps.cooldownMs ?? DEFAULT_COOLDOWN_MS;
  const now = deps.now ?? (() => Date.now());
  const speakEdge = deps.speakEdge ?? defaultSpeakEdge;
  const speakServer = deps.speakServer ?? defaultSpeakServer;

  let lastKey = '';
  let lastTime = 0;

  return {
    reset(): void {
      lastKey = '';
      lastTime = 0;
    },
    async speak(text: string, lang: string, provider: TtsProvider, options?: SpeakOptions): Promise<void> {
      const trimmed = String(text || '').trim();
      if (!trimmed) return;

      if (options?.confidence !== undefined && !canSpeakDetection({ text: trimmed, confidence: options.confidence })) {
        return;
      }

      const key = `${provider}:${lang}:${trimmed}`;
      const t = now();
      if (key === lastKey && t - lastTime < cooldownMs) {
        return;
      }
      lastKey = key;
      lastTime = t;

      if (provider === 'server' || provider === 'elevenlabs') {
        try {
          await speakServer(trimmed, lang);
          return;
        } catch {
          // fall through to edge
        }
      }

      await speakEdge(trimmed, lang);
    },
  };
}
