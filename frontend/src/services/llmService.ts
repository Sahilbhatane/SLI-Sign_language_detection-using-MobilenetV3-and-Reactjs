import axios from 'axios';

/**
 * Calls backend `/llm/correct` (never send API keys from the browser).
 */
export async function correctSentence(text: string, maxTokens = 256): Promise<string> {
  const trimmed = String(text || '').trim();
  if (!trimmed) return '';

  const response = await axios.post(
    '/api/llm/correct',
    { text: trimmed, max_tokens: maxTokens },
    { timeout: 60_000 }
  );

  const corrected = response.data?.corrected;
  return typeof corrected === 'string' && corrected.trim() ? corrected.trim() : trimmed;
}
