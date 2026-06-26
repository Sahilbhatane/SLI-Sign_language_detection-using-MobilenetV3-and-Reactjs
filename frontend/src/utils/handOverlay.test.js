import { describe, expect, it } from 'vitest';
import { HAND_OVERLAY_STYLES, normalizeHandOverlayList } from './handOverlay';

describe('handOverlay', () => {
  it('normalizes multi-hand WebRTC payload', () => {
    const hands = normalizeHandOverlayList({
      hands: [
        { bbox_norm: [0.1, 0.2, 0.3, 0.4], landmarks_norm: [[0.15, 0.25]] },
        { bbox_norm: [0.5, 0.2, 0.7, 0.4], landmarks_norm: [[0.55, 0.25]] },
      ],
    });
    expect(hands).toHaveLength(2);
    expect(hands[0].bbox).toEqual([0.1, 0.2, 0.3, 0.4]);
  });

  it('falls back to legacy single-hand bbox', () => {
    const hands = normalizeHandOverlayList({
      bbox: [0.1, 0.2, 0.3, 0.4],
      landmarks: [[0.15, 0.25]],
    });
    expect(hands).toHaveLength(1);
  });

  it('provides two distinct overlay styles', () => {
    expect(HAND_OVERLAY_STYLES).toHaveLength(2);
    expect(HAND_OVERLAY_STYLES[0].bbox).not.toBe(HAND_OVERLAY_STYLES[1].bbox);
  });
});
