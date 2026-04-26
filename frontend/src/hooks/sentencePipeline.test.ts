import { describe, expect, it } from 'vitest';

import {
  appendBufferDedupe,
  streakReadyForAppend,
  tokenJaccardSimilarity,
  updatePhraseStreak,
} from './sentencePipeline';
import { DETECTING_PHRASE } from '../utils/signSpeech';

describe('sentencePipeline helpers', () => {
  it('updates streak only for accepted phrases with confidence >= 0.95', () => {
    let s = { phrase: null as string | null, count: 0, confidences: [] as number[] };
    s = updatePhraseStreak(s, 'hello', 0.94, 3);
    expect(s).toEqual({ phrase: null, count: 0, confidences: [] });

    s = { phrase: null, count: 0, confidences: [] };
    s = updatePhraseStreak(s, 'hello', 0.95, 3);
    s = updatePhraseStreak(s, 'hello', 0.95, 3);
    s = updatePhraseStreak(s, 'hello', 0.95, 3);
    expect(s.phrase).toBe('hello');
    expect(s.count).toBe(3);
    expect(s.confidences).toEqual([0.95, 0.95, 0.95]);

    s = updatePhraseStreak(s, DETECTING_PHRASE, 1, 3);
    expect(s).toEqual({ phrase: null, count: 0, confidences: [] });
  });

  it('requires average confidence across stable frames >= 0.97', () => {
    let s = { phrase: null as string | null, count: 0, confidences: [] as number[] };
    s = updatePhraseStreak(s, 'hi', 0.95, 3);
    s = updatePhraseStreak(s, 'hi', 0.95, 3);
    s = updatePhraseStreak(s, 'hi', 0.98, 3);
    expect(s.count).toBe(3);
    expect(streakReadyForAppend(s, 3, 0.97)).toBe(false);

    s = { phrase: null, count: 0, confidences: [] };
    s = updatePhraseStreak(s, 'hi', 0.98, 3);
    s = updatePhraseStreak(s, 'hi', 0.97, 3);
    s = updatePhraseStreak(s, 'hi', 0.98, 3);
    expect(streakReadyForAppend(s, 3, 0.97)).toBe(true);
  });

  it('dedupes consecutive identical buffer entries', () => {
    expect(appendBufferDedupe([], 'a')).toEqual(['a']);
    expect(appendBufferDedupe(['a'], 'a')).toEqual(['a']);
    expect(appendBufferDedupe(['a'], 'b')).toEqual(['a', 'b']);
  });

  it('computes token Jaccard similarity', () => {
    expect(tokenJaccardSimilarity('hello world', 'hello world')).toBe(1);
    expect(tokenJaccardSimilarity('a b', 'c d')).toBe(0);
    expect(tokenJaccardSimilarity('I go store', 'I went to the store')).toBeGreaterThan(0.2);
  });
});
