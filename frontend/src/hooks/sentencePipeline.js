import { translateText } from '../services/translationService';
import { DETECTING_PHRASE } from '../utils/signSpeech';

/** @typedef {{ phrase: string; confidence: number }} DetectionLike */

const DEFAULT_STABLE_FRAMES = 5;
const DEFAULT_IDLE_MS = 3000;

/**
 * @param {string} a
 * @param {string} b
 */
export function tokenJaccardSimilarity(a, b) {
  const ta = new Set(
    String(a || '')
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean)
  );
  const tb = new Set(
    String(b || '')
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean)
  );
  if (!ta.size || !tb.size) return 0;
  let inter = 0;
  for (const w of ta) {
    if (tb.has(w)) inter += 1;
  }
  const union = ta.size + tb.size - inter;
  return union ? inter / union : 0;
}

/**
 * Update streak state for same-phrase consecutive frames (phrase accepted; each frame conf >= 0.95).
 * @param {{ phrase: string | null; count: number; confidences?: number[] }} streak
 * @param {string | null} phrase
 * @param {number} confidence01
 * @param {number} required
 */
export function updatePhraseStreak(streak, phrase, confidence01, required = DEFAULT_STABLE_FRAMES) {
  if (!phrase || phrase === DETECTING_PHRASE || confidence01 < 0.95) {
    return { phrase: null, count: 0, confidences: [] };
  }
  if (streak.phrase === phrase) {
    const confidences = [...(streak.confidences || []), confidence01];
    return { phrase, count: streak.count + 1, confidences };
  }
  return { phrase, count: 1, confidences: [confidence01] };
}

/**
 * True when the last `required` frames for the streak have average confidence >= minAvg.
 * @param {{ phrase: string | null; count: number; confidences?: number[] }} streak
 */
export function streakReadyForAppend(streak, required = DEFAULT_STABLE_FRAMES, minAvg = 0.97) {
  if (streak.count < required) return false;
  const slice = (streak.confidences || []).slice(-required);
  if (slice.length < required) return false;
  const avg = slice.reduce((x, y) => x + y, 0) / slice.length;
  return avg >= minAvg;
}

/**
 * @param {string[]} buffer
 * @param {string} phrase
 */
export function appendBufferDedupe(buffer, phrase) {
  const p = String(phrase || '').trim();
  if (!p) return buffer;
  const last = buffer.length ? buffer[buffer.length - 1] : null;
  if (last === p) return buffer;
  return [...buffer, p];
}

/**
 * @param {object} params
 * @param {string[]} params.buffer
 * @param {boolean} params.voiceEnabled
 * @param {{ speak: (t:string,l:string,p:string,o?:object)=>Promise<void> }} params.tts
 * @param {'edge'|'server'|'elevenlabs'} params.ttsProvider
 * @param {boolean} params.speakTranslation
 * @param {string} params.selectedLanguage
 * @param {string} params.translation
 * @param {boolean} params.llmGrammarEnabled
 * @param {(text: string) => Promise<string>} [params.runLlm]
 * @param {(used: boolean) => void} [params.onLlmFollowUp]
 */
export async function finalizeSentenceSpeech({
  buffer,
  voiceEnabled,
  tts,
  ttsProvider,
  speakTranslation,
  selectedLanguage,
  translation,
  llmGrammarEnabled,
  runLlm,
  onLlmFollowUp,
}) {
  if (!voiceEnabled || !buffer.length) return;

  const raw = buffer.join(' ').trim();
  if (!raw) return;

  const TTS_LANG = {
    hi: 'hi-IN',
    mr: 'mr-IN',
    ta: 'ta-IN',
    te: 'te-IN',
    es: 'es-ES',
    fr: 'fr-FR',
    de: 'de-DE',
    ja: 'ja-JP',
    zh: 'zh-CN',
    ar: 'ar-SA',
  };
  const lang =
    speakTranslation && selectedLanguage !== 'en'
      ? TTS_LANG[selectedLanguage] || selectedLanguage
      : 'en-US';

  let textForFirstSpeak = raw;
  if (speakTranslation && selectedLanguage !== 'en') {
    const t = (translation || '').trim();
    if (t && !t.startsWith('⚠️')) {
      textForFirstSpeak = t;
    }
  }

  await tts.speak(textForFirstSpeak, lang, ttsProvider);

  if (llmGrammarEnabled && typeof runLlm === 'function') {
    void (async () => {
      try {
        const corrected = (await runLlm(raw)).trim();
        if (!corrected || corrected === raw) {
          onLlmFollowUp?.(false);
          return;
        }
        const sim = tokenJaccardSimilarity(raw, corrected);
        if (sim < 0.72 || sim >= 0.998) {
          onLlmFollowUp?.(false);
          return;
        }
        let second = corrected;
        if (speakTranslation && selectedLanguage !== 'en') {
          try {
            second = await translateText(corrected, selectedLanguage, 'en');
          } catch {
            second = corrected;
          }
        }
        await tts.speak(second, lang, ttsProvider);
        onLlmFollowUp?.(true);
      } catch {
        onLlmFollowUp?.(false);
      }
    })();
  }
}

export { DEFAULT_STABLE_FRAMES, DEFAULT_IDLE_MS, DETECTING_PHRASE };
