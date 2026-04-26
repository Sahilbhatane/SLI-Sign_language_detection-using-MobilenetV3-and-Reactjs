import { beforeEach, describe, expect, it, vi } from 'vitest';
import { canSpeakDetection, createTtsService } from './ttsService';

describe('ttsService', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-26T10:00:00Z'));
  });

  it('only allows speech for accepted detections at or above 0.95 confidence', () => {
    expect(canSpeakDetection({ text: 'hello', confidence: 0.949 })).toBe(false);
    expect(canSpeakDetection({ text: 'hello', confidence: 0.95 })).toBe(true);
    expect(canSpeakDetection({ text: 'Detecting...', confidence: 1 })).toBe(false);
    expect(canSpeakDetection({ text: '', confidence: 1 })).toBe(false);
  });

  it('suppresses duplicate utterances during the cooldown window', async () => {
    const speakEdge = vi.fn().mockResolvedValue(undefined);
    const service = createTtsService({ speakEdge, cooldownMs: 2000, semanticDedupeMs: 0 });

    await service.speak('hello', 'en-US', 'edge');
    await service.speak('hello', 'en-US', 'edge');

    expect(speakEdge).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(2001);
    await service.speak('hello', 'en-US', 'edge');

    expect(speakEdge).toHaveBeenCalledTimes(2);
  });

  it('falls back to edge speech when server speech fails', async () => {
    const speakEdge = vi.fn().mockResolvedValue(undefined);
    const speakServer = vi.fn().mockRejectedValue(new Error('server down'));
    const service = createTtsService({ speakEdge, speakServer });

    await service.speak('namaste', 'hi-IN', 'server');

    expect(speakServer).toHaveBeenCalledWith('namaste', 'hi-IN');
    expect(speakEdge).toHaveBeenCalledWith('namaste', 'hi-IN');
  });

  it('skips same phrase within semantic dedupe window even after cooldown', async () => {
    const speakEdge = vi.fn().mockResolvedValue(undefined);
    let t = 0;
    const service = createTtsService({
      speakEdge,
      cooldownMs: 500,
      semanticDedupeMs: 10_000,
      now: () => {
        t += 1;
        return t;
      },
    });

    await service.speak('again', 'en-US', 'edge');
    await service.speak('again', 'en-US', 'edge');
    expect(speakEdge).toHaveBeenCalledTimes(1);

    t += 600;
    await service.speak('again', 'en-US', 'edge');
    expect(speakEdge).toHaveBeenCalledTimes(1);

    t += 10_000;
    await service.speak('again', 'en-US', 'edge');
    expect(speakEdge).toHaveBeenCalledTimes(2);
  });
});
