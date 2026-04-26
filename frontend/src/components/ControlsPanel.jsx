import React from 'react';

const ControlsPanel = ({
  voiceEnabled,
  onVoiceEnabledChange,
  speakTranslation,
  onSpeakTranslationChange,
  selectedLanguage,
  ttsProvider,
  onTtsProviderChange,
  offlineMode,
  onOfflineModeChange,
  onExportTranscript,
  speakingActive,
}) => {
  return (
    <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 space-y-4">
      <div className="flex flex-wrap gap-4 items-center">
        <label className="flex items-center gap-2 text-gray-200 text-sm cursor-pointer select-none">
          <input
            type="checkbox"
            className="rounded border-gray-600"
            checked={voiceEnabled}
            onChange={(e) => onVoiceEnabledChange(e.target.checked)}
          />
          Voice Mode ON (≥95% + pipeline)
        </label>

        <label
          className={`flex items-center gap-2 text-sm select-none ${
            selectedLanguage === 'en' ? 'text-gray-500 cursor-not-allowed' : 'text-gray-200 cursor-pointer'
          }`}
        >
          <input
            type="checkbox"
            className="rounded border-gray-600"
            disabled={selectedLanguage === 'en'}
            checked={speakTranslation}
            onChange={(e) => onSpeakTranslationChange(e.target.checked)}
          />
          Speak translation
        </label>

        <label className="flex items-center gap-2 text-gray-200 text-sm cursor-pointer select-none">
          <input
            type="checkbox"
            className="rounded border-gray-600"
            checked={offlineMode}
            onChange={(e) => onOfflineModeChange(e.target.checked)}
          />
          Offline mode (no cloud translate / LLM)
        </label>
      </div>

      <div className="flex flex-wrap gap-4 items-center">
        <label className="flex items-center gap-2 text-gray-200 text-sm select-none">
          <span className="text-gray-400">TTS provider</span>
          <select
            value={ttsProvider}
            onChange={(e) => onTtsProviderChange(e.target.value)}
            className="bg-gray-700 text-white border border-gray-600 rounded-lg px-2 py-1 text-sm"
          >
            <option value="edge">Browser (edge)</option>
            <option value="server">Server /api/tts</option>
            <option value="elevenlabs">Server ElevenLabs path</option>
          </select>
        </label>

        <button
          type="button"
          onClick={onExportTranscript}
          className="px-3 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white text-sm border border-gray-600"
        >
          Export transcript (JSON)
        </button>

        {speakingActive && (
          <div className="flex items-center gap-2 text-xs text-gray-300" aria-live="polite">
            <span className="inline-flex h-2 w-12 rounded bg-gray-600 overflow-hidden">
              <span className="w-1/2 bg-blue-500 animate-pulse" />
            </span>
            Speaking…
          </div>
        )}
      </div>
    </div>
  );
};

export default ControlsPanel;
