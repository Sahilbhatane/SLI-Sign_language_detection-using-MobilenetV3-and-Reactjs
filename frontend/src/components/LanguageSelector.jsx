import React from 'react';
import { motion } from 'framer-motion';

const LanguageSelector = ({ selectedLanguage, onLanguageChange, translation, isTranslating }) => {
  const languages = [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'hi', name: 'Hindi', flag: '🇮🇳' },
    { code: 'mr', name: 'Marathi', flag: '🇮🇳' },
    { code: 'es', name: 'Spanish', flag: '🇪🇸' },
    { code: 'fr', name: 'French', flag: '🇫🇷' },
    { code: 'de', name: 'German', flag: '🇩🇪' },
    { code: 'ja', name: 'Japanese', flag: '🇯🇵' },
    { code: 'zh', name: 'Chinese', flag: '🇨🇳' },
    { code: 'ar', name: 'Arabic', flag: '🇸🇦' },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="text-white font-semibold flex items-center">
          <span className="text-xl mr-2">🌐</span>
          Translate To:
        </label>
      </div>

      {/* Language Dropdown */}
      <div className="relative">
        <select
          value={selectedLanguage}
          onChange={(e) => onLanguageChange(e.target.value)}
          className="w-full bg-gray-700 text-white border border-gray-600 rounded-xl px-4 py-3 pr-10 appearance-none cursor-pointer hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
        >
          {languages.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.flag} {lang.name}
            </option>
          ))}
        </select>
        <div className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none">
          <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Translation Display */}
      {translation && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 rounded-xl p-6 border-l-4 border-blue-500"
        >
          <div className="flex items-start justify-between mb-2">
            <h3 className="text-gray-400 text-sm font-semibold uppercase">Translation</h3>
            {isTranslating && (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                className="text-blue-400"
              >
                ⏳
              </motion.div>
            )}
          </div>
          <motion.p
            key={translation}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-white text-3xl font-semibold"
          >
            {translation}
          </motion.p>
        </motion.div>
      )}

      {/* Language Quick Select Pills */}
      <div className="flex flex-wrap gap-2">
        {languages.slice(0, 5).map((lang) => (
          <motion.button
            key={lang.code}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onLanguageChange(lang.code)}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
              selectedLanguage === lang.code
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {lang.flag} {lang.name}
          </motion.button>
        ))}
      </div>
    </div>
  );
};

export default LanguageSelector;
