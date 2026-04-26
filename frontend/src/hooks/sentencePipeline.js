import { DETECTING_PHRASE } from '../utils/signSpeech';

/** @typedef {{ phrase: string; confidence: number }} DetectionLike */

const DEFAULT_STABLE_FRAMES = 5;
const DEFAULT_IDLE_MS = 3000;

/**
 * Update streak state for same-phrase consecutive frames (phrase must be accepted, not Detecting).
 * @param {{ phrase: string | null; count: number }} streak
 * @param {string | null} phrase
 * @param {number} confidence01
 * @param {number} required
 */
export function updatePhraseStreak(streak, phrase, confidence01, required = DEFAULT_STABLE_FRAMES) {
  if (!phrase || phrase === DETECTING_PHRASE || confidence01 < 0.95) {
    return { phrase: null, count: 0 };
  }
  if (streak.phrase === phrase) {
    return { phrase, count: streak.count + 1 };
  }
  return { phrase, count: 1 };
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

  await tts.speak('Forming sentence', lang, 'edge');

  let text = raw;
  if (llmGrammarEnabled && typeof runLlm === 'function') {
    try {
      text = await runLlm(raw);
    } catch {
      text = raw;
    }
  }

  if (speakTranslation && selectedLanguage !== 'en') {
    const t = (translation || '').trim();
    if (t && !t.startsWith('⚠️')) {
      text = t;
    }
  }

  await tts.speak(text, lang, ttsProvider);
}

export { DEFAULT_STABLE_FRAMES, DEFAULT_IDLE_MS, DETECTING_PHRASE };
