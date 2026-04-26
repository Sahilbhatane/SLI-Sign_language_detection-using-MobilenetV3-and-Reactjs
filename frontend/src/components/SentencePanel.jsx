import React from 'react';

const SentencePanel = ({ rawSentence, buffer, isForming, sentenceTranslation, selectedLanguage }) => {
  const hasBuffer = buffer && buffer.length > 0;

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-3">
      <h3 className="text-white font-semibold flex items-center gap-2">
        <span className="text-xl">📝</span>
        Sentence builder
      </h3>

      {isForming && (
        <div className="text-amber-300 text-sm font-medium animate-pulse">Forming sentence…</div>
      )}

      <div>
        <p className="text-gray-400 text-xs uppercase mb-1">Live buffer</p>
        <p className="text-white text-lg min-h-[1.75rem]">{hasBuffer ? buffer.join(' · ') : '—'}</p>
      </div>

      <div>
        <p className="text-gray-400 text-xs uppercase mb-1">Joined sentence</p>
        <p className="text-gray-200">{rawSentence || '—'}</p>
      </div>

      {selectedLanguage !== 'en' && (
        <div>
          <p className="text-gray-400 text-xs uppercase mb-1">Sentence translation</p>
          <p className="text-blue-200">{sentenceTranslation || '—'}</p>
        </div>
      )}
    </div>
  );
};

export default SentencePanel;
