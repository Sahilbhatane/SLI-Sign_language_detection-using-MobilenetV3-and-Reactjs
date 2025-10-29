import axios from 'axios';

// LibreTranslate API - you can use a public instance or self-host
const LIBRETRANSLATE_API = 'https://libretranslate.com/translate';

/**
 * Translate text using LibreTranslate API
 * @param {string} text - Text to translate
 * @param {string} targetLang - Target language code (e.g., 'hi', 'mr', 'es')
 * @param {string} sourceLang - Source language code (default: 'en')
 * @returns {Promise<string>} - Translated text
 */
export const translateText = async (text, targetLang, sourceLang = 'en') => {
  try {
    // If target language is English, return original text
    if (targetLang === 'en') {
      return text;
    }

    const response = await axios.post(
      LIBRETRANSLATE_API,
      {
        q: text,
        source: sourceLang,
        target: targetLang,
        format: 'text'
      },
      {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 10000 // 10 second timeout
      }
    );

    if (response.data && response.data.translatedText) {
      return response.data.translatedText;
    }

    throw new Error('Translation failed: No translated text in response');
  } catch (error) {
    console.error('Translation error:', error);
    
    // Fallback error messages
    if (error.code === 'ECONNABORTED') {
      throw new Error('Translation timeout - please try again');
    }
    
    if (error.response?.status === 429) {
      throw new Error('Translation rate limit exceeded - please wait');
    }
    
    if (error.response?.status === 400) {
      throw new Error('Unsupported language pair');
    }
    
    throw new Error(error.message || 'Translation failed');
  }
};

/**
 * Batch translate multiple texts
 * @param {string[]} texts - Array of texts to translate
 * @param {string} targetLang - Target language code
 * @param {string} sourceLang - Source language code
 * @returns {Promise<string[]>} - Array of translated texts
 */
export const batchTranslate = async (texts, targetLang, sourceLang = 'en') => {
  try {
    const translations = await Promise.all(
      texts.map(text => translateText(text, targetLang, sourceLang))
    );
    return translations;
  } catch (error) {
    console.error('Batch translation error:', error);
    throw error;
  }
};

/**
 * Get supported languages from LibreTranslate
 * @returns {Promise<Array>} - Array of supported language objects
 */
export const getSupportedLanguages = async () => {
  try {
    const response = await axios.get('https://libretranslate.com/languages');
    return response.data;
  } catch (error) {
    console.error('Error fetching languages:', error);
    // Return default languages if API fails
    return [
      { code: 'en', name: 'English' },
      { code: 'hi', name: 'Hindi' },
      { code: 'mr', name: 'Marathi' },
      { code: 'es', name: 'Spanish' },
      { code: 'fr', name: 'French' },
      { code: 'de', name: 'German' },
      { code: 'ja', name: 'Japanese' },
      { code: 'zh', name: 'Chinese' },
      { code: 'ar', name: 'Arabic' },
    ];
  }
};

export default {
  translateText,
  batchTranslate,
  getSupportedLanguages
};
