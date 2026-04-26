import { useEffect, useRef, useState } from 'react';

import { createTtsService, type TtsProvider } from '../services/ttsService';
import { translateText } from '../services/translationService';
import { DETECTING_PHRASE } from '../utils/signSpeech';

import {
  appendBufferDedupe,
  finalizeSentenceSpeech,
  updatePhraseStreak,
  DEFAULT_IDLE_MS,
  DEFAULT_STABLE_FRAMES,
} from './sentencePipeline';

export type DetectionPayload = {
  phrase: string;
  confidence: number;
  timestamp?: string;
};

export function useSentencePipeline({
  detection,
  voiceEnabled,
  speakTranslation,
  selectedLanguage,
  translation,
  isTranslating,
  ttsProvider,
  llmGrammarEnabled,
  correctSentence,
  offlineMode,
}: {
  detection: DetectionPayload | null;
  voiceEnabled: boolean;
  speakTranslation: boolean;
  selectedLanguage: string;
  translation: string;
  isTranslating: boolean;
  ttsProvider: TtsProvider;
  llmGrammarEnabled: boolean;
  correctSentence?: (text: string) => Promise<string>;
  offlineMode: boolean;
}) {
  const [buffer, setBuffer] = useState<string[]>([]);
  const [isForming, setIsForming] = useState(false);
  const [sentenceTranslation, setSentenceTranslation] = useState('');

  const streakRef = useRef({ phrase: null as string | null, count: 0 });
  const bufferRef = useRef<string[]>([]);

  const ttsRef = useRef<ReturnType<typeof createTtsService> | null>(null);
  if (!ttsRef.current) {
    ttsRef.current = createTtsService();
  }

  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const optsRef = useRef({
    voiceEnabled,
    speakTranslation,
    selectedLanguage,
    ttsProvider,
    llmGrammarEnabled,
    correctSentence,
    isTranslating,
    translation,
  });
  optsRef.current = {
    voiceEnabled,
    speakTranslation,
    selectedLanguage,
    ttsProvider,
    llmGrammarEnabled,
    correctSentence,
    isTranslating,
    translation,
    offlineMode,
  };

  const clearIdle = () => {
    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }
  };

  const armIdle = () => {
    clearIdle();
    idleTimerRef.current = setTimeout(() => {
      void (async () => {
        const snap = bufferRef.current.slice();
        if (snap.length === 0) return;

        const o = optsRef.current;
        setIsForming(true);

        let translatedForSpeech = '';
        const rawJoined = snap.join(' ').trim();
        if (o.offlineMode || o.selectedLanguage === 'en') {
          translatedForSpeech = rawJoined;
          setSentenceTranslation(translatedForSpeech);
        } else {
          try {
            translatedForSpeech = await translateText(rawJoined, o.selectedLanguage, 'en');
            setSentenceTranslation(translatedForSpeech);
          } catch {
            translatedForSpeech = '';
            setSentenceTranslation('');
          }
        }

        await finalizeSentenceSpeech({
          buffer: snap,
          voiceEnabled: o.voiceEnabled,
          tts: ttsRef.current!,
          ttsProvider: o.ttsProvider,
          speakTranslation: o.speakTranslation,
          selectedLanguage: o.selectedLanguage,
          translation: translatedForSpeech,
          llmGrammarEnabled: o.llmGrammarEnabled && !o.offlineMode,
          runLlm: o.correctSentence,
        });

        setBuffer(() => {
          bufferRef.current = [];
          return [];
        });
        streakRef.current = { phrase: null, count: 0 };
        setIsForming(false);
      })();
    }, DEFAULT_IDLE_MS);
  };

  useEffect(() => {
    if (!voiceEnabled) {
      clearIdle();
      ttsRef.current?.reset();
      streakRef.current = { phrase: null, count: 0 };
      setBuffer(() => {
        bufferRef.current = [];
        return [];
      });
      setIsForming(false);
      setSentenceTranslation('');
    }
  }, [voiceEnabled]);

  useEffect(() => {
    bufferRef.current = buffer;
  }, [buffer]);

  useEffect(() => {
    if (!detection?.phrase) return;

    const phrase = detection.phrase;
    const conf = typeof detection.confidence === 'number' ? detection.confidence : 0;

    armIdle();

    if (phrase === DETECTING_PHRASE) {
      streakRef.current = { phrase: null, count: 0 };
      return;
    }

    streakRef.current = updatePhraseStreak(
      streakRef.current,
      phrase,
      conf,
      DEFAULT_STABLE_FRAMES
    );

    if (streakRef.current.count !== DEFAULT_STABLE_FRAMES || conf < 0.95) {
      return;
    }

    setBuffer((prev) => {
      const next = appendBufferDedupe(prev, phrase);
      bufferRef.current = next;
      return next;
    });

    const o = optsRef.current;
    if (!o.voiceEnabled) return;

    void (async () => {
      const TTS_LANG: Record<string, string> = {
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
        o.speakTranslation && o.selectedLanguage !== 'en'
          ? TTS_LANG[o.selectedLanguage] || o.selectedLanguage
          : 'en-US';

      let text = phrase;
      if (o.speakTranslation && o.selectedLanguage !== 'en') {
        if (o.isTranslating) return;
        const t = (o.translation || '').trim();
        if (t && !t.startsWith('⚠️')) text = t;
      }

      await ttsRef.current!.speak(text, lang, o.ttsProvider, { confidence: conf });
    })();
  }, [detection]);

  useEffect(() => () => clearIdle(), []);

  const rawSentence = buffer.length ? buffer.join(' ') : '';

  return {
    buffer,
    rawSentence,
    isForming,
    sentenceTranslation,
  };
}
