import { useEffect, useRef } from 'react';
import { computeVoiceTrigger, speakText, DETECTING_PHRASE } from '../utils/signSpeech';

const initialState = {
  streak: 0,
  candidate: null,
  lastSpokenText: null,
  speakNow: false,
  textToSpeak: null,
};

/**
 * Speaks stable detections when voice is enabled.
 */
export function useSignVoice({
  voiceEnabled,
  speakTranslation,
  selectedLanguage,
  detection,
  translation,
  isTranslating,
}) {
  const stateRef = useRef({ ...initialState });
  /** Last phrase+text we passed to speechSynthesis (avoid duplicate calls). */
  const utteranceKeyRef = useRef(null);

  useEffect(() => {
    if (!voiceEnabled) {
      stateRef.current = { ...initialState };
      utteranceKeyRef.current = null;
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
      return;
    }

    const phrase = detection?.phrase;
    if (!phrase || phrase === DETECTING_PHRASE) {
      stateRef.current = computeVoiceTrigger(stateRef.current, phrase || DETECTING_PHRASE);
      utteranceKeyRef.current = null;
      return;
    }

    const next = computeVoiceTrigger(stateRef.current, phrase);
    stateRef.current = next;

    if (!next.speakNow || !next.textToSpeak) return;

    let text = next.textToSpeak;
    if (speakTranslation && selectedLanguage !== 'en') {
      if (isTranslating) return;
      const t = (translation || '').trim();
      if (t && !t.startsWith('⚠️')) text = t;
    }

    const TTS_LANG = {
      hi: 'hi-IN',
      mr: 'mr-IN',
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
    const utteranceKey = `${phrase}::${text}::${lang}`;
    if (utteranceKeyRef.current === utteranceKey) return;
    utteranceKeyRef.current = utteranceKey;

    speakText(text, { lang });

    stateRef.current = {
      ...next,
      lastSpokenText: phrase,
    };
  }, [
    voiceEnabled,
    speakTranslation,
    selectedLanguage,
    detection,
    translation,
    isTranslating,
  ]);

  useEffect(() => {
    return () => {
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);
}
