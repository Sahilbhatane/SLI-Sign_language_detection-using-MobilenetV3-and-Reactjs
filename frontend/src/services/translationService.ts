import axios from 'axios';

type GlossMap = Record<string, Record<string, string>>;

let glossCache: GlossMap | null = null;

async function loadGloss(): Promise<GlossMap> {
  if (glossCache) return glossCache;
  try {
    const res = await axios.get('/gloss/isl_gloss.json', { timeout: 5000 });
    glossCache = (res.data && typeof res.data === 'object' ? res.data : {}) as GlossMap;
  } catch {
    glossCache = {};
  }
  return glossCache;
}

/** Normalized lookup keys: lowercase, snake_case, and compact forms. */
export function glossKeyVariants(text: string): string[] {
  const raw = String(text || '').trim().toLowerCase();
  if (!raw) return [];
  const snake = raw.replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '');
  const compact = raw.replace(/\s+/g, '').replace(/[^a-z0-9]/g, '');
  return [...new Set([raw, snake, compact].filter(Boolean))];
}

export async function getLocalGlossTranslation(text: string, targetLang: string): Promise<string | null> {
  const g = await loadGloss();
  for (const key of glossKeyVariants(text)) {
    const row = g[key];
    if (!row) continue;
    const hit = row[targetLang];
    if (typeof hit === 'string' && hit.trim()) return hit.trim();
  }
  return null;
}

/**
 * Translate text: local gloss map first, then backend `/translate`, then original on failure.
 */
export async function translateText(text: string, targetLang: string, sourceLang = 'en'): Promise<string> {
  if (targetLang === 'en') {
    return text;
  }

  const local = await getLocalGlossTranslation(text, targetLang);
  if (local) return local;

  try {
    const response = await axios.post(
      '/api/translate',
      {
        q: text,
        source: sourceLang,
        target: targetLang,
        format: 'text',
      },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 15_000,
      }
    );

    if (response.data && typeof response.data.translatedText === 'string') {
      return response.data.translatedText;
    }

    return text;
  } catch {
    return text;
  }
}

export async function batchTranslate(texts: string[], targetLang: string, sourceLang = 'en'): Promise<string[]> {
  return Promise.all(texts.map((t) => translateText(t, targetLang, sourceLang)));
}

export async function getSupportedLanguages(): Promise<Array<{ code: string; name: string }>> {
  try {
    const response = await axios.get('/api/translate/languages', { timeout: 10_000 });
    return response.data;
  } catch {
    return [
      { code: 'en', name: 'English' },
      { code: 'hi', name: 'Hindi' },
      { code: 'mr', name: 'Marathi' },
      { code: 'ta', name: 'Tamil' },
      { code: 'te', name: 'Telugu' },
    ];
  }
}
