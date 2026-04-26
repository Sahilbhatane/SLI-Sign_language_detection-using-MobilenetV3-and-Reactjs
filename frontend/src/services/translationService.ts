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

export async function getLocalGlossTranslation(text: string, targetLang: string): Promise<string | null> {
  const key = String(text || '').trim().toLowerCase();
  if (!key) return null;
  const g = await loadGloss();
  const row = g[key];
  if (!row) return null;
  const hit = row[targetLang];
  return typeof hit === 'string' && hit.trim() ? hit.trim() : null;
}

/**
 * Translate text. Prefer backend proxy `/translate` (same-origin) to avoid CORS and keep config server-side.
 */
export async function translateText(text: string, targetLang: string, sourceLang = 'en'): Promise<string> {
  if (targetLang === 'en') {
    return text;
  }

  const local = await getLocalGlossTranslation(text, targetLang);
  if (local) return local;

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

  throw new Error('Translation failed: No translated text in response');
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
