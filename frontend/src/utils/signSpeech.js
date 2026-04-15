/**
 * Stable phrase detection for TTS: require consecutive identical accepted phrases.
 * Resets when input is empty or "Detecting...".
 */

export const DETECTING_PHRASE = 'Detecting...';

export function computeVoiceTrigger(prev, phrase, options = {}) {
  const requiredStreak = options.requiredStreak ?? 2;
  const detecting = !phrase || phrase === DETECTING_PHRASE;

  if (detecting) {
    return {
      streak: 0,
      candidate: null,
      lastSpokenText: null,
      speakNow: false,
      textToSpeak: null,
    };
  }

  if (phrase !== prev.candidate) {
    const speakNow = requiredStreak === 1 && phrase !== prev.lastSpokenText;
    return {
      streak: 1,
      candidate: phrase,
      lastSpokenText: prev.lastSpokenText,
      speakNow,
      textToSpeak: speakNow ? phrase : null,
    };
  }

  const streak = prev.streak + 1;
  const stable = streak >= requiredStreak;
  const speakNow = stable && phrase !== prev.lastSpokenText;

  return {
    streak,
    candidate: phrase,
    lastSpokenText: prev.lastSpokenText,
    speakNow,
    textToSpeak: speakNow ? phrase : null,
  };
}

/**
 * Speak text using Web Speech API (browser).
 */
export function speakText(text, options = {}) {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  const trimmed = String(text || '').trim();
  if (!trimmed) return;

  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(trimmed);
  u.lang = options.lang || 'en-US';
  u.rate = typeof options.rate === 'number' ? options.rate : 1.0;
  u.pitch = typeof options.pitch === 'number' ? options.pitch : 1.0;
  window.speechSynthesis.speak(u);
}
