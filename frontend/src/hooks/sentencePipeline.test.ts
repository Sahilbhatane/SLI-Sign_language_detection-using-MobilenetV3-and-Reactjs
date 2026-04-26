import { describe, expect, it } from 'vitest';

import { appendBufferDedupe, updatePhraseStreak } from './sentencePipeline';
import { DETECTING_PHRASE } from '../utils/signSpeech';

describe('sentencePipeline helpers', () => {
  it('updates streak only for accepted phrases with confidence >= 0.95', () => {
    let s = { phrase: null as string | null, count: 0 };
    s = updatePhraseStreak(s, 'hello', 0.94, 3);
    expect(s).toEqual({ phrase: null, count: 0 });

    s = { phrase: null, count: 0 };
    s = updatePhraseStreak(s, 'hello', 0.95, 3);
    s = updatePhraseStreak(s, 'hello', 0.95, 3);
    s = updatePhraseStreak(s, 'hello', 0.95, 3);
    expect(s).toEqual({ phrase: 'hello', count: 3 });

    s = updatePhraseStreak(s, DETECTING_PHRASE, 1, 3);
    expect(s).toEqual({ phrase: null, count: 0 });
  });

  it('dedupes consecutive identical buffer entries', () => {
    expect(appendBufferDedupe([], 'a')).toEqual(['a']);
    expect(appendBufferDedupe(['a'], 'a')).toEqual(['a']);
    expect(appendBufferDedupe(['a'], 'b')).toEqual(['a', 'b']);
  });
});
