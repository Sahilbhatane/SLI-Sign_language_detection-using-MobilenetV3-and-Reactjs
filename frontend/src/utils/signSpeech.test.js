import { describe, it, expect } from 'vitest';
import { computeVoiceTrigger, DETECTING_PHRASE } from './signSpeech';

const empty = {
  streak: 0,
  candidate: null,
  lastSpokenText: null,
  speakNow: false,
  textToSpeak: null,
};

describe('computeVoiceTrigger', () => {
  it('resets on Detecting...', () => {
    const prev = { streak: 2, candidate: 'hello', lastSpokenText: 'x', speakNow: false, textToSpeak: null };
    const out = computeVoiceTrigger(prev, DETECTING_PHRASE);
    expect(out.streak).toBe(0);
    expect(out.candidate).toBe(null);
    expect(out.lastSpokenText).toBe(null);
    expect(out.speakNow).toBe(false);
  });

  it('emits after two consecutive identical phrases', () => {
    let s = { ...empty };
    s = computeVoiceTrigger(s, 'hello');
    expect(s.speakNow).toBe(false);
    s = computeVoiceTrigger(s, 'hello');
    expect(s.speakNow).toBe(true);
    expect(s.textToSpeak).toBe('hello');
  });

  it('does not emit same phrase twice without detecting gap', () => {
    let s = { ...empty };
    s = computeVoiceTrigger(s, 'hello');
    s = computeVoiceTrigger(s, 'hello');
    expect(s.speakNow).toBe(true);
    s = { ...s, lastSpokenText: 'hello' };
    s = computeVoiceTrigger(s, 'hello');
    expect(s.speakNow).toBe(false);
  });

  it('resets streak when phrase changes', () => {
    let s = { ...empty };
    s = computeVoiceTrigger(s, 'a');
    s = computeVoiceTrigger(s, 'b');
    expect(s.streak).toBe(1);
    expect(s.candidate).toBe('b');
  });
});
