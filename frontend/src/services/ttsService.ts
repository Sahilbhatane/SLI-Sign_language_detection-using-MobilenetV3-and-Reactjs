import axios from 'axios';

import { DETECTING_PHRASE } from '../utils/signSpeech';

export type TtsProvider = 'edge' | 'server' | 'elevenlabs';

export type SpeakOptions = {
  /** When set, speech is allowed only if confidence >= SPEAK_MIN_CONFIDENCE. */
  confidence?: number;
};

const DEFAULT_COOLDOWN_MS = 1500;
/** Same phrase is not re-spoken within this window even after cooldown expires. */
const DEFAULT_SEMANTIC_DEDUPE_MS = 8000;

/**
 * Minimum confidence to voice a detection.
 *
 * Must be reachable: the classifier is trained with label_smoothing=0.1, which
 * caps the correct-class probability near ~0.90, so a 0.95 gate made voice output
 * effectively never fire. This is aligned with the detection-acceptance threshold
 * (backend `min_confidence`, default 0.6) so anything shown to the user can also be
 * spoken. Sentence-level stability (consecutive identical frames) is enforced
 * separately in the sentence pipeline, so this is a secondary safety filter.
 */
export const SPEAK_MIN_CONFIDENCE = 0.6;

export function canSpeakDetection(input: { text: string; confidence: number }): boolean {
  const t = String(input.text || '').trim();
  if (!t || t === DETECTING_PHRASE) return false;
  return input.confidence >= SPEAK_MIN_CONFIDENCE;
}

export type TtsServiceDeps = {
  speakEdge?: (text: string, lang: string) => Promise<void>;
  speakServer?: (text: string, lang: string) => Promise<void>;
  cooldownMs?: number;
  semanticDedupeMs?: number;
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

function semanticPhraseKey(text: string, lang: string, provider: string): string {
  return `${provider}|${lang}|${String(text).trim().toLowerCase()}`;
}

export function createTtsService(deps: TtsServiceDeps = {}) {
  const cooldownMs = deps.cooldownMs ?? DEFAULT_COOLDOWN_MS;
  const semanticDedupeMs = deps.semanticDedupeMs ?? DEFAULT_SEMANTIC_DEDUPE_MS;
  const now = deps.now ?? (() => Date.now());
  const speakEdge = deps.speakEdge ?? defaultSpeakEdge;
  const speakServer = deps.speakServer ?? defaultSpeakServer;

  let lastKey = '';
  let lastTime = 0;
  const lastSpokenPhraseAt = new Map<string, number>();

  return {
    reset(): void {
      lastKey = '';
      lastTime = 0;
      lastSpokenPhraseAt.clear();
    },
    async speak(text: string, lang: string, provider: TtsProvider, options?: SpeakOptions): Promise<void> {
      const trimmed = String(text || '').trim();
      if (!trimmed) return;

      if (
        options?.confidence !== undefined &&
        !canSpeakDetection({ text: trimmed, confidence: options.confidence })
      ) {
        return;
      }

      const phraseKey = semanticPhraseKey(trimmed, lang, provider);
      const t = now();
      const lastPhrase = lastSpokenPhraseAt.get(phraseKey);
      if (
        semanticDedupeMs > 0 &&
        lastPhrase !== undefined &&
        t - lastPhrase < semanticDedupeMs
      ) {
        return;
      }

      const key = `${provider}:${lang}:${trimmed}`;
      if (key === lastKey && t - lastTime < cooldownMs) {
        return;
      }
      lastKey = key;
      lastTime = t;

      if (provider === 'server' || provider === 'elevenlabs') {
        try {
          await speakServer(trimmed, lang);
          lastSpokenPhraseAt.set(phraseKey, now());
          return;
        } catch {
          // fall through to edge
        }
      }

      await speakEdge(trimmed, lang);
      lastSpokenPhraseAt.set(phraseKey, now());
    },
  };
}
