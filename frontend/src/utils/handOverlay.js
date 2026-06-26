/**
 * Normalize hand overlay payloads from REST or WebRTC into a list of hands to draw.
 * @param {object | null | undefined} source
 * @returns {{ bbox: number[], landmarks: [number, number][] | null }[]}
 */
export function normalizeHandOverlayList(source) {
  if (!source) return [];

  if (Array.isArray(source.hands) && source.hands.length > 0) {
    return source.hands
      .map((h) => ({
        bbox: Array.isArray(h.bbox_norm) ? h.bbox_norm : h.bbox,
        landmarks: Array.isArray(h.landmarks_norm) ? h.landmarks_norm : h.landmarks ?? null,
      }))
      .filter((h) => Array.isArray(h.bbox) && h.bbox.length === 4);
  }

  if (Array.isArray(source.bbox) && source.bbox.length === 4) {
    return [
      {
        bbox: source.bbox,
        landmarks: Array.isArray(source.landmarks) ? source.landmarks : null,
      },
    ];
  }

  return [];
}

/** Cyan + pink styles for hand 1 and hand 2. */
export const HAND_OVERLAY_STYLES = [
  { bbox: 'rgba(34, 211, 238, 0.95)', skel: 'rgba(52, 211, 153, 0.85)', joint: 'rgba(250, 250, 250, 0.95)' },
  { bbox: 'rgba(244, 114, 182, 0.95)', skel: 'rgba(251, 191, 36, 0.85)', joint: 'rgba(255, 255, 255, 0.95)' },
];
